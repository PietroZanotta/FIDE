"""Transfer-inclusive native-versus-JAX Galerkin chunk benchmark."""
from __future__ import annotations
import argparse, json, statistics, time
import jax, jax.numpy as jnp, numpy as np
from mfsi.galerkin_tesseract import assemble_galerkin_chunk_tesseract_forward
jax.config.update("jax_enable_x64", True)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--samples",type=int,default=256)
    p.add_argument("--basis",type=int,default=280); p.add_argument("--particles",type=int,default=16)
    p.add_argument("--dimensions",type=int,default=2); p.add_argument("--repetitions",type=int,default=3); a=p.parse_args()
    key=jax.random.PRNGKey(17); v=jax.random.normal(key,(a.samples,a.basis),dtype=jnp.float64)
    g=jax.random.normal(jax.random.fold_in(key,1),(a.samples,a.basis,a.particles,a.dimensions),dtype=jnp.float64)
    w=jax.nn.softmax(jax.random.normal(jax.random.fold_in(key,2),(a.samples,),dtype=jnp.float64))
    h=jax.random.normal(jax.random.fold_in(key,3),(a.samples,),dtype=jnp.float64)
    fn=jax.jit(lambda v,g,w,h:(jnp.einsum("n,njpd,nkpd->jk",w,g,g),jnp.einsum("n,n,nk->k",w,h,v),jnp.einsum("n,nk->k",w,v),jnp.einsum("n,n->",w,h)))
    ref=fn(v,g,w,h); ref[0].block_until_ready(); assemble_galerkin_chunk_tesseract_forward(*map(np.asarray,(v,g,w,h)))
    nt=[]; jt=[]
    for _ in range(a.repetitions):
        t=time.perf_counter(); native=assemble_galerkin_chunk_tesseract_forward(*[np.asarray(x) for x in (v,g,w,h)]); nt.append(time.perf_counter()-t)
        t=time.perf_counter(); ref=fn(v,g,w,h); ref[0].block_until_ready(); jt.append(time.perf_counter()-t)
    error=max(float(np.max(np.abs(native[n]-np.asarray(x)))) for n,x in zip(("gram","raw_load","basis_mean","forcing_sum"),ref,strict=True))
    print(json.dumps({"shape":[a.samples,a.basis,a.particles,a.dimensions],"native_transfer_inclusive_seconds":statistics.median(nt),"jax_seconds":statistics.median(jt),"jax_over_native_speedup":statistics.median(jt)/statistics.median(nt),"maximum_absolute_discrepancy":error},indent=2))
if __name__=="__main__": main()

