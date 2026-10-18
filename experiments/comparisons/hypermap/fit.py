"""Final fixed-protocol full-source HyperMap fit; no hyperparameter search."""
from pathlib import Path
import hashlib,json,math,time,traceback
import higher
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from . import helpers as H
COMMON_PATH=Path(H.__file__)

def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def fit(args,Net,model_path,data,sources,pdim,scale,chosen,dest):
    H.seed_all(args.seed)
    model=Net(976,pdim,64,256).to(args.device)
    initial_hash=state_hash(model)
    opt=torch.optim.Adam(model.parameters(),lr=H.OUTER_LR)
    ctx_rng=np.random.RandomState(args.seed)
    loaders={}
    for cell in sources:
        gen=torch.Generator().manual_seed(args.seed+1111+H.CELLS.index(cell))
        loaders[cell]=DataLoader(TensorDataset(torch.tensor(data[cell]['ti'])),
            batch_size=H.TRAIN_BATCH,shuffle=True,drop_last=True,generator=gen)
    contract={'method':'HyperMap-L5 post-selection full-source refit','target':args.target,'seed':args.seed,
        'protocol':'fixed source-selected full-source fit',
        'selected_inner_lr':chosen['inner_lr'],'fixed_epochs':chosen['best_epoch'],
        'selection_score_from_prior_source_validation':chosen['best_source_score'],
        'selection_sha256':args.selection_sha256,'refit_validation':False,'hyperparameter_search':False,
        'source_only_frozen_selection':True,'target_query_labels_used_for_selection':False,
        'fresh_initialization':True,'initial_state_sha256':initial_hash,
        'sources':sources,'full_source_response_rows_used':True,'scale':scale,
        'scaling_uses_only_allowed_full_source_responses':True,
        'meta_lr':H.OUTER_LR,'batch':H.TRAIN_BATCH,'drop_last':True,
        'meta_tasks':H.META_TASKS,'inner_batches':H.INNER_BATCHES,'inner_steps_per_batch':H.INNER_STEPS,
        'second_order':True,'copy_initial_weights':False,'gradient_clip':None,
        'optimizer':'Adam outer, SGD inner','early_stopping':False,
        'loss':'0.5 * mean_batch(sum_genes((prediction - truth)^2))',
        'failure_policy':'mark failed; no learning-rate change, fallback checkpoint, skipped updates or automatic rescue',
        'official_model_sha256':H.sha(model_path),'official_model_path':str(model_path),
        'runner_sha256':H.sha(__file__),'frozen_helper_sha256':H.sha(COMMON_PATH),
        'input':'one-hot drug identity and constant zero signed-Level5 control',
        'native_HyperMap_reproduction':False,'original_batch_softmax_preserved':True,
        'prediction_batching':'fixed query order; max512; duplicate a singleton remainder then crop',
        'target_adaptation':'same 20 frozen anchors and 3 support-loss backtracking steps',
        'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__}
    H.write(dest/'refit_contract.json',contract)
    start=time.monotonic();history=[];effective=0;error=None
    torch.cuda.reset_peak_memory_stats() if str(args.device).startswith("cuda") else None
    try:
        for epoch in range(1,chosen['best_epoch']+1):
            begin=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
            tasks=list(ctx_rng.choice(sources,H.META_TASKS,replace=True));losses=[];draws=[]
            for cell in tasks:
                iterator=iter(loaders[cell]);d=data[cell];used=[]
                def batch():
                    nonlocal iterator
                    try:indices=next(iterator)[0]
                    except StopIteration:
                        iterator=iter(loaders[cell]);indices=next(iterator)[0]
                    used.extend(indices.tolist());indices=indices.to(args.device)
                    return H.features(d,indices,pdim),d['y'][indices]
                with higher.innerloop_ctx(model,torch.optim.SGD(model.parameters(),lr=chosen['inner_lr']),
                    copy_initial_weights=False,track_higher_grads=True) as (fnet,diffopt):
                    for _ in range(H.INNER_BATCHES):
                        x,y=batch();zero=torch.zeros(len(x),976,device=args.device)
                        for _ in range(H.INNER_STEPS):
                            pred=fnet(zero,x)[0];loss=.5*(pred-y).square().sum(1).mean()
                            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite inner loss epoch{epoch} context{cell}')
                            diffopt.step(loss)
                    x,y=batch();outer=.5*(fnet(torch.zeros(len(x),976,device=args.device),x)[0]-y).square().sum(1).mean()
                    if not torch.isfinite(outer):raise FloatingPointError(f'nonfinite outer loss epoch{epoch} context{cell}')
                    (outer/H.META_TASKS).backward();losses.append(float(outer.detach()))
                draws.append(hashlib.sha256(np.asarray(used,dtype=np.int64).tobytes()).hexdigest())
                del fnet,diffopt,x,y,zero,pred,loss,outer
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f'nonfinite meta-gradient epoch{epoch}')
            grad_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))
            if not math.isfinite(grad_norm):raise FloatingPointError(f'nonfinite meta-gradient norm epoch{epoch}')
            opt.step();effective+=1
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError(f'nonfinite updated model parameters epoch{epoch}')
            history.append({'epoch':epoch,'effective_updates':effective,'meta_loss':float(np.mean(losses)),
                'task_losses':losses,'task_contexts':tasks,'training_index_hashes':draws,
                'meta_gradient_norm':grad_norm,'training_epoch_seconds':time.monotonic()-begin,
                'elapsed_seconds':time.monotonic()-start})
            H.write(dest/'refit_history.json',history)
            print(json.dumps({'phase':'refit','target':args.target,'seed':args.seed,'epoch':epoch,
                'fixed_epochs':chosen['best_epoch'],'inner_lr':chosen['inner_lr'],
                'meta_loss':history[-1]['meta_loss'],'elapsed_seconds':history[-1]['elapsed_seconds']}),flush=True)
    except (FloatingPointError,RuntimeError) as e:
        error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    result={'target':args.target,'seed':args.seed,'valid':error is None,'finite':error is None,
        'effective_updates':effective,'fixed_epochs':chosen['best_epoch'],'inner_lr':chosen['inner_lr'],
        'elapsed_seconds':time.monotonic()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30 if str(args.device).startswith("cuda") else 0.0,
        'early_stopping':False,'skipped_updates':[],'automatic_rescue':False,'error':error,
        'initial_state_sha256':initial_hash,'selection_sha256':args.selection_sha256,
        'target_query_metrics_computed':False,'runner_sha256':H.sha(__file__)}
    if error is None:
        assert effective==chosen['best_epoch']
        torch.save({'state_dict':model.state_dict(),'contract':contract,'epoch':effective},dest/'source_refit.pt')
        result['checkpoint_sha256']=H.sha(dest/'source_refit.pt')
        result['final_state_sha256']=state_hash(model)
    H.write(dest/'REFIT_TRAINING_COMPLETE.json',result)
    if error is not None:
        H.write(dest/'FOLD_FAILED.json',{'complete':False,'phase':'training','details':result})
        return None,result
    return model,result

def predict_target(args,model,genes,vocab,scale,query,anchors,result,dest):
    lookup={n:i for i,n in enumerate(vocab)};pdim=len(vocab)
    hx=torch.tensor([lookup[n] for n in query],device=args.device)
    pred0=H.predict(model,hx,pdim)*scale
    # The selected hyperparameters and complete source refit are already frozen.
    # Only now are target responses materialized for adaptation and evaluation.
    with np.load(args.data/'sources'/f'{args.target}.npz',allow_pickle=False) as z:
        nmap={str(n):i for i,n in enumerate(z['chemical_names'])}
        ai=np.array([nmap[n] for n in anchors]);qi=np.array([nmap[n] for n in query])
        anchor_truth=z['chemical_native'][ai].astype(np.float32);truth=z['chemical_native'][qi].astype(np.float32)
    x=torch.nn.functional.one_hot(torch.tensor([lookup[n] for n in anchors],device=args.device),pdim).float()
    adapted,steps=H.safe_adapt(model,x,torch.tensor(anchor_truth/scale,device=args.device),args.seed+83471)
    pred20=H.predict(adapted,hx,pdim)*scale
    for budget,pred in [(0,pred0),(20,pred20)]:
        out=dest/f'D{budget}';out.mkdir(parents=True,exist_ok=True)
        assert np.isfinite(pred).all()
        np.savez_compressed(out/'predictions.npz',names=query,genes=genes,truth=truth,pred=pred,
            anchor_names=anchors if budget else np.array([],dtype=str))
        H.write(out/'COMPLETE.json',{'complete':True,'method':f'HyperMap-D{budget}',
            'target':args.target,'seed':args.seed,'budget':budget,'queries':len(query),'genes':len(genes),
            'query_names_seen_in_source_training':len(query),'selected_inner_lr':result['inner_lr'],
            'fixed_refit_epochs':result['fixed_epochs'],'effective_updates':result['effective_updates'],
            'source_selection_sha256':args.selection_sha256,'source_checkpoint_sha256':result['checkpoint_sha256'],
            'prediction_sha256':H.sha(out/'predictions.npz'),'target_metrics_computed_here':False,'smoke':args.smoke,
            'native_HyperMap_reproduction':False,'refit_hyperparameters_frozen_before_target':True})
    torch.save({'state_dict':adapted.state_dict(),'source_checkpoint_sha256':result['checkpoint_sha256'],
        'anchor_names':anchors,'adaptation_steps':steps},dest/'D20/adapted.pt')
    H.write(dest/'target_adaptation_log.json',steps)
    H.write(dest/'FOLD_COMPLETE.json',{'complete':True,'target':args.target,'seed':args.seed,
        'inner_lr':result['inner_lr'],'fixed_epochs':result['fixed_epochs'],
        'effective_updates':result['effective_updates'],'source_selection_sha256':args.selection_sha256,
        'query_names_seen_in_source_training':len(query),'target_metrics_computed_here':False,'smoke':args.smoke,
        'runner_sha256':H.sha(__file__)})
