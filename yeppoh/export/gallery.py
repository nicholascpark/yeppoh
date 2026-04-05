"""WebXR gallery generator — creates a Three.js scene for VR viewing.

Generates an HTML file that loads exported .glb meshes into a
VR-ready gallery scene. Works in Meta Quest browser via WebXR.

This is Phase 4+ — the gallery is a static HTML file that can be
served from localhost and opened in any WebXR-capable browser.
"""

from __future__ import annotations

from pathlib import Path


GALLERY_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>yeppoh — Living Sculpture Gallery</title>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.170/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170/examples/jsm/"
        }}
    }}
    </script>
    <style>
        body {{ margin: 0; background: #000; }}
        canvas {{ display: block; }}
    </style>
</head>
<body>
<script type="module">
import * as THREE from 'three';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
import {{ VRButton }} from 'three/addons/webxr/VRButton.js';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 1.6, 3);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);

// Lighting
const ambient = new THREE.AmbientLight(0x404040, 0.5);
scene.add(ambient);

const spot = new THREE.SpotLight(0xffffff, 2, 20, Math.PI / 6);
spot.position.set(0, 5, 2);
spot.castShadow = true;
scene.add(spot);

const rim = new THREE.PointLight(0x4488ff, 1, 10);
rim.position.set(-2, 2, -1);
scene.add(rim);

// Ground
const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({{ color: 0x111111 }})
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// Load sculptures
const loader = new GLTFLoader();
const sculptures = {sculpture_paths};

sculptures.forEach((path, i) => {{
    const angle = (i / sculptures.length) * Math.PI * 2;
    const radius = 2;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;

    loader.load(path, (gltf) => {{
        const model = gltf.scene;
        model.position.set(x, 1, z);
        model.scale.set(2, 2, 2);

        // Pedestal
        const pedestal = new THREE.Mesh(
            new THREE.CylinderGeometry(0.3, 0.35, 0.8, 32),
            new THREE.MeshStandardMaterial({{ color: 0x222222, metalness: 0.8 }})
        );
        pedestal.position.set(x, 0.4, z);
        scene.add(pedestal);

        scene.add(model);
    }});
}});

// Animation loop
renderer.setAnimationLoop(() => {{
    controls.update();
    renderer.render(scene, camera);
}});

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


def generate_gallery(
    mesh_paths: list[str | Path],
    output_path: str | Path = "gallery.html",
) -> Path:
    """Generate a WebXR gallery HTML file.

    Args:
        mesh_paths: list of paths to .glb mesh files
        output_path: where to save the HTML

    Returns:
        Path to the generated HTML file
    """
    output_path = Path(output_path)

    # Convert paths to relative strings for the HTML
    path_strs = [f'"{Path(p).name}"' for p in mesh_paths]
    paths_js = f"[{', '.join(path_strs)}]"

    html = GALLERY_TEMPLATE.replace("{sculpture_paths}", paths_js)

    output_path.write_text(html)
    print(f"Gallery generated → {output_path}")
    print(f"Serve with: python -m http.server 8080 --directory {output_path.parent}")
    print(f"Open Quest browser → http://<your-ip>:8080/{output_path.name}")

    return output_path
