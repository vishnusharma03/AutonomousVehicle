# generate_visuals.py
import pygame
import numpy as np
from math import ceil, sqrt

def visualize(vm, sensor_names, width=1200, height=800, sync=False):
    """
    Fetches latest sensor data from the VehicleManager (vm) and displays all
    selected sensors in a single pygame window grid.
    """
    pygame.init()
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Vehicle Sensor Visualization')
    clock = pygame.time.Clock()

    n = len(sensor_names)
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))
    cell_w = width // cols
    cell_h = height // rows

    running = True
    world = vm.world
    while running:
        if sync:
            world.tick()
        else:
            world.wait_for_tick()

        win.fill((0, 0, 0))
        for idx, name in enumerate(sensor_names):
            data = vm.get_latest_sensor_data(name)
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h

            if data is None:
                font = pygame.font.Font(None, 24)
                text = font.render(f"{name}: waiting...", True, (255, 255, 255))
                win.blit(text, (x + 10, y + 10))
            else:
                if name == 'rgb_camera':
                    surf = pygame.surfarray.make_surface(data.swapaxes(0, 1))
                    surf = pygame.transform.scale(surf, (cell_w, cell_h))
                    win.blit(surf, (x, y))
                elif name == 'lidar':
                    points = data[:, :2]
                    pts = points.copy()
                    pts -= pts.min(axis=0)
                    pts /= pts.ptp(axis=0)
                    pts *= [cell_w - 1, cell_h - 1]
                    for px, py in pts.astype(int):
                        win.set_at((x + px, y + py), (255, 255, 255))
                else:
                    font = pygame.font.Font(None, 20)
                    lines = []
                    if hasattr(data, 'raw_data'):
                        lines = [f"raw: {len(data.raw_data)} bytes"]
                    else:
                        try:
                            lines = [f"{k}: {v:.4f}" for k, v in data.__dict__.items()]
                        except:
                            lines = [str(data)]
                    for i, line in enumerate(lines):
                        txt = font.render(line, True, (255, 255, 255))
                        win.blit(txt, (x + 5, y + 5 + i * 18))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        clock.tick(20)

    pygame.quit()