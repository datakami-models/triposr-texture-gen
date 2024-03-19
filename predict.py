# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

WEIGHTS_FOLDER = "/src/models/"
os.environ['HF_HOME'] = WEIGHTS_FOLDER 
os.environ['HF_HUB_CACHE'] = WEIGHTS_FOLDER
os.environ['HUGGINGFACE_HUB_CACHE'] = WEIGHTS_FOLDER
os.environ['TORCH_HOME'] = WEIGHTS_FOLDER
os.environ['TORCH_HUB'] = WEIGHTS_FOLDER
os.environ['TRANSFORMERS_CACHE'] = WEIGHTS_FOLDER
os.environ['XDG_CACHE_HOME'] = WEIGHTS_FOLDER # dirty hack for diffusers 0.26.3

import sys
import shutil
import shlex
import subprocess
import open3d as o3d
from PIL import Image
from cog import BasePredictor, Input, Path
import numpy as np

from text2texture import process_tripo_mesh, raycast_mesh, ray_hits_to_depth, write_mesh, set_tmesh_tex, compute_texture


ACCEPTED_EXTENSIONS = ['.glb','.obj']

def compute_raycast_texture(tmesh, raycast_result, rgb_im, size=512):
    print()
    print('computing UV atlas for', len(tmesh.triangle.indices), 'triangles')
    tmesh.compute_uvatlas(size, parallel_partitions=2)
    
    print()
    print('generating texture')
    imdata = tmesh.bake_vertex_attr_textures(size, {'colors'})['colors'].numpy()

    imdata = (imdata * 255).astype('u1')
    prim_ids = raycast_result['primitive_ids'].numpy().flatten()
    mask = prim_ids != 0xffff_ffff
    return compute_texture(
        tmesh,
        raycast_result['primitive_uvs'].numpy().reshape(-1, 2)[mask],
        prim_ids[mask],
        np.array(rgb_im).reshape(-1, 3)[mask],
        size,
        imdata,
    )

def text2texture(mesh, desc, steps, depth_txt2img_path, img_model, device, out_path_base):
    print()
    print('processing mesh')
    mesh = process_tripo_mesh(mesh)

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    raycast_result = raycast_mesh(tmesh)
    depth_im = ray_hits_to_depth(raycast_result)
    depth_path = f'{out_path_base}-preproc-depth.png'

    print('saving depth map at', depth_path)
    depth_im.save(depth_path)

    painted_path = f'{out_path_base}-preproc-depth-paint.png'
    depth_paint_args = [
        depth_txt2img_path,
        desc, 
        depth_path,
        painted_path,
        '--steps',
        str(steps),
        '--image-model',
        img_model,
        *(['--device', device] if device else [])
    ]

    print()
    print('>', *(shlex.quote(arg) for arg in depth_paint_args))
    subprocess.run(
        [sys.executable, *depth_paint_args],
        check=True,
        env=os.environ | {'PYTORCH_ENABLE_MPS_FALLBACK': '1'},
    )

    tex_imdata = compute_raycast_texture(tmesh, raycast_result, Image.open(painted_path))

    set_tmesh_tex(tmesh, tex_imdata)
    return tmesh

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        mesh: Path = Input(description="Output mesh from TripoSR"),
        mesh_format: str = Input(description="Filetype of the output mush from TripoSR", choices=ACCEPTED_EXTENSIONS, default='.glb'),
        prompt: str = Input(description="textual description of the desired appearance"),
        steps: int = Input(description="Num inference steps for texture image gen", default=12, ge=1, le=1000),
                            
    ) -> Path:
        """Run a single prediction on the model"""
        
        if not mesh.suffix in ACCEPTED_EXTENSIONS:
            mesh_filename_with_extension = f"{mesh}{mesh_format}"
            shutil.copyfile(mesh, mesh_filename_with_extension)
            mesh = mesh_filename_with_extension
        
        output_dir = "output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        triangle_mesh = o3d.io.read_triangle_mesh(str(mesh))
        out_base = os.path.join(output_dir, 'mesh')
        image_model = 'Lykon/dreamshaper-8'
        tmesh = text2texture(
            mesh=triangle_mesh,
            desc=prompt,
            steps=steps,
            depth_txt2img_path=os.path.join(os.path.dirname(__file__), 'depth_txt2img.py'),
            img_model=image_model,
            device="",
            out_path_base=out_base,
        )

        out_mesh_base = f'{out_base}-tex'
        # print('writing new mesh to', f'{out_mesh_base}.obj')
        write_mesh(out_mesh_base, tmesh)
        
        return Path(f'{out_mesh_base}.obj')
