#include <zlib.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <filesystem>

// A bounded-memory gene-by-cell CSV scanner. Only requested gene rows are retained.
// Every expressed gene contributes to each cell's library size, including rows
// absent from the selected panel. Zlib supports the source's concatenated members.
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"usage: stream_counts counts.csv.gz wanted_symbols.txt outdir\n";return 2;}
 std::filesystem::create_directories(argv[3]); std::string out=argv[3];
 std::unordered_set<std::string>wanted;std::ifstream wf(argv[2]);std::string s;
 if(!wf)throw std::runtime_error("cannot open requested gene-symbol list");
 while(std::getline(wf,s))if(!s.empty())wanted.insert(s);
 gzFile f=gzopen(argv[1],"rb");if(!f)throw std::runtime_error("cannot open counts");
 gzbuffer(f,8*1024*1024);std::vector<char>buf(8*1024*1024);
 bool header=true,labelmode=true,keep=false,mito=false,havevalue=false;
 std::string label,headertext;size_t ncell=0,col=0,rows=0,nkeep=0;uint64_t value=0;
 std::vector<uint64_t>total,mt;std::vector<uint32_t>detected,idx,vals;
 std::ofstream index(out+"/retained_rows.tsv");index<<"row_id\tfeature\tsymbol\tfile\tnnz\n";
 auto start=std::chrono::steady_clock::now();
 auto consume=[&](){
  if(!havevalue)throw std::runtime_error("empty count token row "+std::to_string(rows));
  if(col>=ncell)throw std::runtime_error("too many cell columns");
  total[col]+=value;if(value){detected[col]++;if(mito)mt[col]+=value;if(keep){idx.push_back(col);vals.push_back(value);}}
  col++;value=0;havevalue=false;
 };
 int n;
 while((n=gzread(f,buf.data(),buf.size()))>0){
  for(int j=0;j<n;j++){
   char c=buf[j];if(c=='\r')continue;
   if(header){
    if(c=='\n'){
     ncell=std::count(headertext.begin(),headertext.end(),',');
     std::ofstream hf(out+"/counts_header.csv");hf<<headertext<<'\n';hf.close();
     total.assign(ncell,0);mt.assign(ncell,0);detected.assign(ncell,0);
     header=false;std::cerr<<"cells="<<ncell<<std::endl;headertext.clear();headertext.shrink_to_fit();
    }else headertext+=c;
    continue;
   }
   if(labelmode){
    if(c==','){
     size_t a=label.find(':');size_t b=label.find(':',a+1);
     std::string symbol=(a==std::string::npos)?label:label.substr(a+1,b-a-1);
     keep=wanted.count(symbol);mito=symbol.rfind("MT-",0)==0;
     labelmode=false;col=0;idx.clear();vals.clear();
    }else label+=c;
    continue;
   }
   if(c>='0'&&c<='9'){value=value*10+(c-'0');havevalue=true;}
   else if(c==','||c=='\n'){
    consume();
    if(c=='\n'){
     if(col!=ncell)throw std::runtime_error("column count mismatch "+std::to_string(rows));
     if(keep){
      size_t a=label.find(':');size_t b=label.find(':',a+1);std::string symbol=a==std::string::npos?label:label.substr(a+1,b-a-1);
      std::string name="row_"+std::to_string(rows)+".bin";
      std::ofstream of(out+"/"+name,std::ios::binary);uint64_t nnz=idx.size();of.write((char*)&nnz,8);
      of.write((char*)idx.data(),nnz*4);of.write((char*)vals.data(),nnz*4);
      index<<rows<<'\t'<<label<<'\t'<<symbol<<'\t'<<name<<'\t'<<nnz<<'\n';nkeep++;
     }
     rows++;label.clear();labelmode=true;
     if(rows%1000==0){
      double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      std::cerr<<"rows="<<rows<<" retained="<<nkeep<<" compressed_GB="<<gzoffset(f)/1e9<<" seconds="<<elapsed<<std::endl;
      index.flush();
     }
    }
   }else throw std::runtime_error("non-integer count character");
  }
 }
 if(n<0)throw std::runtime_error(gzerror(f,nullptr));
 if(!labelmode||!label.empty())throw std::runtime_error("unexpected incomplete final row");
 gzclose(f);index.close();
 auto save=[&](std::string fn,auto &v){std::ofstream of(out+"/"+fn,std::ios::binary);of.write((char*)v.data(),v.size()*sizeof(v[0]));};
 save("total_umi.uint64",total);save("mitochondrial_umi.uint64",mt);save("n_detected.uint32",detected);
 std::ofstream meta(out+"/scan_summary.json");meta<<"{\"n_cells\":"<<ncell<<",\"n_gene_rows\":"<<rows<<",\"n_retained_rows\":"<<nkeep<<",\"complete\":true}\n";
 std::cerr<<"COMPLETE rows="<<rows<<" retained="<<nkeep<<std::endl;
}
