
# import carla

# # Connect to the CARLA server
# client = carla.Client('212.2.246.117', 2000)

# def cleanup_all_actors(client):
#     """Destroy all vehicles and sensors in the world."""
#     world = client.get_world()
#     all_actors = world.get_actors()
#     targets = list(all_actors.filter('vehicle.*')) + list(all_actors.filter('sensor.*'))
#     if targets:
#         cmds = [carla.command.DestroyActor(actor) for actor in targets]
#         client.apply_batch(cmds)

# if __name__ == '__main__':
#     cleanup_all_actors(client)

import carla

# Connect to CARLA
client = carla.Client('212.2.246.117', 2000)
client.set_timeout(10.0)

def cleanup_all_actors(client):
    """Destroy all vehicles and sensors."""
    world = client.get_world()
    actors = world.get_actors()
    targets = list(actors.filter('vehicle.*')) + list(actors.filter('sensor.*'))
    if targets:
        cmds = [carla.command.DestroyActor(a) for a in targets]
        client.apply_batch(cmds)

def reset_world_settings(client):
    """Restore original world settings if synchronous mode was set."""
    world = client.get_world()
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        world.apply_settings(settings)


if __name__ == '__main__':
    # 1. Clean actors (vehicles & sensors)
    cleanup_all_actors(client)

    # 2. Reset world settings (sync off)
    reset_world_settings(client)

    print("Cleanup complete. World restored and Traffic Manager shut down.")
