import taichi as ti
from . import config
from . import global_var

@ti.data_oriented
class Gui():
    def __init__(self):
        self.window = ti.ui.Window("Fluid Simulation", config.gui.res, vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()
        self.camera.position(*config.gui.camera_pos)
        self.camera.lookat(*config.gui.camera_lookat)
        self.camera.fov(config.gui.camera_fov)
        self.canvas.set_background_color(config.gui.background_color)
        self.ambient_color = (0.7, 0.7, 0.7)

        # Toggles
        self.show_cross_section = config.gui.show_cross_section
        self.show_help = True
        self.refresh_window = True

        # stores particles to be shown
        max_n = config.sph.max_particle_count
        self.max_particle_count = max_n
        self.particle_buffer = ti.Struct.field({
            "position": ti.types.vector(3, float),
            "color": ti.types.vector(3, float),
        }, max_n)

        # for color display
        
        self.display_fields = config.gui.display_fields
        self.display_field = self.display_fields[0]
        self.loadColorConfig(self.display_field)
        self.display_field_id = 0

    def loadColorConfig(self, name):
        if not name in config.gui.color_config:
            name = "default"
        print("load color config:", name)
        self.slider_min = config.gui.color_config[name]["slider_min"]
        self.slider_max = config.gui.color_config[name]["slider_max"]
        self.display_min = config.gui.color_config[name]["display_min"]
        self.display_max = config.gui.color_config[name]["display_max"]
            

    def showHelp(self):
        if self.show_help:
            self.window.GUI.begin("options", 0.05, 0.3, 0.2, 0.3)
            self.window.GUI.text("h: help")
            self.window.GUI.text("w: front")
            self.window.GUI.text("s: back")
            self.window.GUI.text("a: left")
            self.window.GUI.text("d: right")
            self.window.GUI.text("RMB: rotate")
            self.window.GUI.text("v: display cross-section")
            self.window.GUI.text("r: run system")
            self.window.GUI.text("f: write file")
            self.window.GUI.text("c: refresh window")
            self.window.GUI.text("o: display field: previous")
            self.window.GUI.text("p: display field: next")
            self.window.GUI.text("n: save timestep snapshots")
            self.window.GUI.text("k: load previous snapshot")
            self.window.GUI.text("l: load next snapshot")
            self.window.GUI.end()
        
    
    def handleInput(self):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        if self.window.get_event(ti.ui.PRESS):
            # run
            if self.window.event.key == 'r':
                global_var.run_sph = not global_var.run_sph
                print("running SPH:", global_var.run_sph)

            if self.window.event.key == 'f':
                global_var.write_file = not global_var.write_file
                print("write file:", global_var.write_file)

            if self.window.event.key == 'v':
                self.show_cross_section = not self.show_cross_section
                print("show cross-section:", self.show_cross_section)

            if self.window.event.key == 'h':
                self.show_help = not self.show_help
                print("show help:", self.show_help)
            
            if self.window.event.key == 'c':
                self.refresh_window = not self.refresh_window
                print("refresh window:", self.refresh_window)
            
            if self.window.event.key == 'n':
                global_var.write_snap_file = not global_var.write_snap_file
                print("write snap file:", global_var.write_snap_file)
            
            if self.window.event.key == 'k':
                if global_var.write_snap_file:
                    self.load_prev_snap = True
                    global_var.run_sph = False
                    print("load prev snap")
            
            if self.window.event.key == 'l':
                if global_var.write_snap_file:
                    self.load_next_snap = True
                    global_var.run_sph = False
                    print("load next snap")
            
            if self.window.event.key == 'o':
                self.display_field_id -= 1
                self.display_field_id %= len(self.display_fields)
                self.display_field = self.display_fields[self.display_field_id]
                print("load prev display field")
                self.loadColorConfig(self.display_field)
            
            if self.window.event.key == 'p':
                self.display_field_id += 1
                self.display_field_id %= len(self.display_fields)
                self.display_field = self.display_fields[self.display_field_id]
                print("load next display field")
                self.loadColorConfig(self.display_field)
    
    def clearTriggerInput(self):
        self.load_prev_snap = False
        self.load_next_snap = False

    @ti.kernel
    def addParticlesKernel(self, particles: ti.template(), colorScheme:ti.template(), field:ti.template(), min_v: float, max_v: float):
        for i in range(particles.particle_count[None]):
            self.particle_buffer.position[i] = particles.position[i]
            self.particle_buffer.color[i] = colorScheme(field, i, min_v, max_v)
        for i in range(particles.particle_count[None],self.max_particle_count):
            self.particle_buffer.position[i] = ti.Vector([-99.0,-99.0,-99.0]) #put other particles away (scene.particles doesn't seem to support sparse fields)
            self.particle_buffer.color[i] = ti.Vector([0.0, 0.0, 0.0])

    def addParticles(self, particles, colorScheme, radius=config.gui.radius):
        self.addParticlesKernel(particles, colorScheme, getattr(particles,self.display_field), self.display_min, self.display_max)
        self.scene.particles(self.particle_buffer.position, per_vertex_color=self.particle_buffer.color, radius=radius)

    @ti.kernel
    def addParticlesCrossSectionKernel(self, particles: ti.template(), colorScheme:ti.template(), field:ti.template(), min_v: float, max_v: float):
        for i in range(particles.particle_count[None]):
            if particles.position[i][0] <= 0.0:
                self.particle_buffer.position[i] = particles.position[i]
                self.particle_buffer.color[i] = colorScheme(field, i, min_v, max_v)
            else:
                self.particle_buffer.position[i] = ti.Vector([-99.0,-99.0,-99.0]) #put other particles away (scene.particles doesn't seem to support sparse fields)
                self.particle_buffer.color[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(particles.particle_count[None],self.max_particle_count):
            self.particle_buffer.position[i] = ti.Vector([-99.0,-99.0,-99.0]) #put other particles away (scene.particles doesn't seem to support sparse fields)
            self.particle_buffer.color[i] = ti.Vector([0.0, 0.0, 0.0])

    def addParticlesCrossSection(self, particles, colorScheme, radius=config.gui.radius):
        self.addParticlesCrossSectionKernel(particles, colorScheme, getattr(particles,self.display_field), self.display_min, self.display_max)
        self.scene.particles(self.particle_buffer.position, per_vertex_color=self.particle_buffer.color, radius=radius)

    def showInputWindow(self):
        gui = self.window.GUI
        gui.begin("color display", 0.05, 0.05, 0.3, 0.2)
        old_min = self.display_min
        old_max = self.display_max
        gui.text("displaying field: " + self.display_field)
        self.display_min = gui.slider_float("Min", old_min, self.slider_min, self.slider_max)
        self.display_max = gui.slider_float("Max", old_max, self.slider_min, self.slider_max)

    def show(self):
        self.showHelp()
        self.showInputWindow()
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.ambient_color)
        self.scene.point_light(pos=(2, 1.5, -1.5), color=(0.8, 0.8, 0.8))
        self.canvas.scene(self.scene)  # Render the scene
        if global_var.write_file:
            self.window.write_image(global_var.root_path + r"\png\frame_" + str(global_var.frame) + ".png")
        self.window.show()