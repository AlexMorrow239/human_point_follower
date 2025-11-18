import numpy as np
import random
import os

from scipy.spatial.transform import Rotation as R

from rospkg import RosPack; rp = RosPack()
repo_path = rp.get_path('rc_isaac_worlds')
usd_repo_path = rp.get_path('usd')

import omni
from omni.isaac.kit import SimulationApp

import argparse
import json

sim_app_config_default = {
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,
    "renderer": "RayTracedLighting"
}
sim_app_config_default_str = json.dumps(sim_app_config_default)

parser = argparse.ArgumentParser(description="HSR Isaac World Setup")
parser.add_argument("--sim_app_config", type=str, default=sim_app_config_default_str, help="Configuration of isaac sim app.")
parser.add_argument("--auto_neutral", action="store_true", help="Auto-send HSR to neutral via MoveIt after spawn.")
args, _ = parser.parse_known_args()

sim_app = SimulationApp(json.loads(args.sim_app_config))
sim_app.set_setting("/app/extensions/installUntrustedExtensions", True)

from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.isaac.core import World
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils import prims, viewports
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.sensor import Camera
from pxr import PhysxSchema, Gf, UsdGeom, Usd, Sdf, UsdPhysics
from omni import usd
from omni.isaac.core.objects import DynamicCuboid
import omni.isaac.core.utils.prims as prim_utils

# Your packages
import robocanes_hsr
from isaac_robot_behavior_start import isaac_robot_behavior_start
from isaac_robot_pose_pub import isaac_robot_pose_pub

class robocanes_isaac_world:
    def init_sim_world(self):
        self.sim_world = World(stage_units_in_meters=1.0)
        self.usd_context = omni.usd.get_context()
        self.stage = self.usd_context.get_stage()

        # Physics Scene (units, gravity, fixed step)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        physics_scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/World/physicsScene"))
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
        physx_api = PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/World/physicsScene"))
        physx_api.CreateTimeStepsPerSecondAttr().Set(120)
        physx_api.CreateEnableCCDAttr().Set(True)

    def _enforce_single_physics_scene(self):
        scenes = [p for p in self.stage.Traverse() if p.GetTypeName() == "PhysicsScene"]
        keep_path = Sdf.Path("/World/physicsScene")
        for p in scenes:
            if p.GetPath() != keep_path:
                p.SetActive(False)
        main = self.stage.GetPrimAtPath(str(keep_path))
        if main and main.IsValid():
            api = PhysxSchema.PhysxSceneAPI.Apply(main)
            api.CreateTimeStepsPerSecondAttr().Set(120)

    def _tune_hsrb_articulation_iters(self):
        root = self.stage.GetPrimAtPath("/World/hsrb")
        if root and root.IsValid():
            art_api = PhysxSchema.PhysxArticulationAPI.Apply(root)
            art_api.CreateSolverPositionIterationCountAttr(4)
            art_api.CreateSolverVelocityIterationCountAttr(1)

    def add_ground_plane(self):
        self.sim_world.scene.add_ground_plane(size=1000, z_position=0.0, color=np.array([1, 1, 1]))

    def add_fancy_cube(self):
        # random x,y ? [2.5, 15.0], keep z=2.0
        rand_x = random.uniform(-2.5, -5.0)
        rand_y = random.uniform(3.0, 8.0)

        self.sim_world.scene.add(
            DynamicCuboid(
                prim_path="/World/Props/Cube",
                name="fancy_cube",
                position=np.array([-2.5, 4.5, 0.7]),
                scale=np.array([0.1, 0.1, 0.1]),
                color=np.array([0, 1.0, 0]),
                size=1.0
            )
        )
        # Ensure cube is un-damped & unclamped
        cube = self.stage.GetPrimAtPath("/World/Cube")
        if cube and cube.IsValid():
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube)
            rb.CreateLinearDampingAttr(0.0)
            rb.CreateAngularDampingAttr(0.0)
            rb.CreateDisableGravityAttr(False)
            rb.CreateMaxLinearVelocityAttr(51.0)

    def add_coffee_table(self):
        coffee_table_usd_path = os.path.join(usd_repo_path, 'robocanes_lab', 'robocanes_lab', 'coffeeTable.usd')

        coffee_table_name = "coffee_table"
        coffee_table_prim_path = f'/World/Props/{coffee_table_name}'

        coffee_table_prim = prims.create_prim(
            prim_path=coffee_table_prim_path,
            usd_path=coffee_table_usd_path,
            translation=[-2.5, 4.5, 0.1],
            orientation=euler_angles_to_quat([0.0, 0.0, 0.0]),
            scale=[1, 1, 1],
            semantic_label=coffee_table_name
        )

    
    def add_scene_lights(self):
        prim_utils.create_prim(
            "/World/Lights/Light_1", "SphereLight",
            position=np.array([1.0, 2.0, 2.0]),
            attributes={
                "inputs:radius": 0.01, 
                "inputs:intensity": 5e3, 
                "inputs:color": (1.0, 1.0, 1.0),  
                "inputs:exposure": 12.0
            }
        )
        prim_utils.create_prim(
            "/World/Lights/Light_2", "SphereLight",
            position=np.array([-10.0, 2.0, 2.0]),
            attributes={
                "inputs:radius": 0.01, 
                "inputs:intensity": 5e4, 
                "inputs:color": (1.0, 1.0, 1.0),  
                "inputs:exposure": 12.0
            }
        )
        prim_utils.create_prim(
            "/World/Lights/Light_3", "SphereLight",
            position=np.array([10.0, 2.0, 2.0]),
            attributes={
                "inputs:radius": 0.01, 
                "inputs:intensity": 5e3, 
                "inputs:color": (1.0, 1.0, 1.0),  
                "inputs:exposure": 12.0
        }
        )
        prim_utils.create_prim(
            "/World/Lights/Light_4", "SphereLight",
            position=np.array([1.0, 10.0, 2.0]),
            attributes={"inputs:radius": 0.01, "inputs:intensity": 5e4, "inputs:color": (1.0, 1.0, 1.0),  "inputs:exposure": 12.0}
        )
        prim_utils.create_prim(
            "/World/Lights/Light_5", "SphereLight",
            position=np.array([1.0, -10.0, 2.0]),
            attributes={"inputs:radius": 0.01, "inputs:intensity": 5e4, "inputs:color": (1.0, 1.0, 1.0),  "inputs:exposure": 12.0}
        )
        prim_utils.create_prim(
            "/World/Lights/Light_6", "SphereLight",
            position=np.array([-2.5, 4.5, 1.25]),
            attributes={
                "inputs:radius": 0.01, 
                "inputs:intensity": 5e4, 
                "inputs:color": (1.0, 1.0, 1.0),  
                "inputs:exposure": 6.0
            }
        )


    def add_hsr(self, robot_name, robot_spawn_position, robot_spawn_orientation):
        self.hsr_instance = robocanes_hsr.hsr(
            prefix=f'/{robot_name}',
            spawn_config={'translation': robot_spawn_position, 'orientation': robot_spawn_orientation, 'scale': [1, 1, 1]}
        )
        self.hsr_instance.onsimulationstart(self.sim_world)
        self.isaac_robot_behavior_start = isaac_robot_behavior_start()
        self.isaac_robot_pose_pub = isaac_robot_pose_pub()

        # Fix invalid mass/inertia on base_footprint if present
        base = self.stage.GetPrimAtPath("/World/hsrb/base_footprint")
        if base and base.IsValid():
            try:
                mass_api = UsdPhysics.MassAPI.Apply(base)
                mass_api.CreateMassAttr(20.0)  # example kg
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.2, 0.2, 0.2))  # example kgï¿½mï¿½
            except Exception as e:
                print(f"[Mass/Inertia patch] Skipped or failed: {e}")

    def add_human_character(self, character_name="F_Business_02", 
                            position=None, rotation=None):
        """
        Add a human character to the scene, positioned to face the HSR robot.
        
        Args:
            character_name: Name of character USD file (without .usd extension)
                        Examples: "male_adult_police_04", "female_adult_police_02",
                                "male_adult_construction_03", etc.
            position: [x, y, z] position. If None, places 2m in front of robot
            rotation: Quaternion [w, x, y, z]. If None, faces toward robot
        """
        import carb.settings
        from isaacsim.core.utils import nucleus
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        
        # Get asset root path for Isaac Sim 4.5
        try:
            asset_root = nucleus.get_assets_root_path()
        except Exception as e:
            print(f"[Warning] Could not use nucleus.get_assets_root_path(): {e}")
            settings = carb.settings.get_settings()
            asset_root = settings.get("/persistent/isaac/asset_root/default")
            if not asset_root:
                asset_root = settings.get("/persistent/isaac/asset_root/cloud")
        
        # Build path to character
        character_usd_path = f"{asset_root}/Isaac/People/Characters/{character_name}/{character_name}.usd"
        
        # Default position: 2 meters in front of robot (robot at [0, 0, 0.01])
        if position is None:
            position = [2.0, 0.0, 0.0]
        
        # Default rotation: facing the robot (180 degrees around Z axis)
        if rotation is None:
            euler = [0.0, 0.0, np.pi]  # 180 degrees yaw
            rotation = euler_angles_to_quat(euler)
        
        # Create the character prim
        character_prim_path = f"/World/Characters/{character_name}"
        character_scale = [1.0, 1.0, 1.0]
        
        print(f"[Human Character] Loading: {character_name}")
        print(f"[Human Character] USD Path: {character_usd_path}")
        print(f"[Human Character] Position: {position}")
        
        character_prim = prims.create_prim(
            prim_path=character_prim_path,
            usd_path=character_usd_path,
            translation=position,
            orientation=rotation,
            scale=character_scale,
            semantic_label=character_name
        )
        
        print(f"[Human Character] Added at: {character_prim_path}")
        
        return character_prim
    
    def populate_sim_world(self):
        self.add_ground_plane()
        # self.add_fancy_cube()
        # self.add_coffee_table()
        self.add_scene_lights()
        self.add_human_character()
        self.add_hsr(robot_name='hsrb', robot_spawn_position=[0, 0, 0.01], robot_spawn_orientation=[0, 0, 0])
        self._enforce_single_physics_scene()
        self._tune_hsrb_articulation_iters()

    def modify_perspective_viewport_camera(self, eye_pos, target_pos):
        viewports.set_camera_view(eye=eye_pos, target=target_pos)
        prim = self.stage.GetPrimAtPath('/OmniverseKit_Persp')
        prim.GetAttribute('horizontalAperture').Set(43.0)
        prim.GetAttribute('focalLength').Set(18.0) 

    def __init__(self):
        self.init_sim_world()
        self.populate_sim_world()
        self.modify_perspective_viewport_camera(
            eye_pos=np.array([1.0, 0.6, 1.8]),
            target_pos=np.array([-4.5, 4.5, 0.0])
        )

if __name__ == '__main__':
    world = robocanes_isaac_world()
    world.sim_world.play()

    # Match physics TPS (120) with multiple substeps per render to avoid slow-motion
    PHYS_TPS = 120
    RENDER_FPS = 30
    SUBSTEPS = max(1, PHYS_TPS // RENDER_FPS)

    while sim_app.is_running():
        for _ in range(SUBSTEPS - 1):
            world.sim_world.step(render=False)
        world.sim_world.step(render=True)
        # read-only per frame:
        world.hsr_instance.step()

    sim_app.close()