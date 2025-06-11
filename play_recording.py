#!/usr/bin/env python3

import carla
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser(description='Play back CARLA simulation recording')
    parser.add_argument('recording_file', type=str, help='Path to the recording file (.log)')
    parser.add_argument('--host', type=str, default='212.2.246.117', help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port (default: 2000)')
    parser.add_argument('--timeout', type=float, default=10.0, help='Client timeout in seconds (default: 10.0)')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds (default: 0.0)')
    parser.add_argument('--duration', type=float, default=0.0, help='Duration to play in seconds (0 = full recording)')
    
    args = parser.parse_args()
    
    try:
        # Connect to CARLA server
        print(f"Connecting to CARLA server at {args.host}:{args.port}")
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        
        # Get world
        world = client.get_world()
        print(f"Connected to world: {world.get_map().name}")
        
        # Get recording info
        try:
            info = client.show_recorder_file_info(args.recording_file, True)
            print(f"Recording info:\n{info}")
        except Exception as e:
            print(f"Warning: Could not get recording info: {e}")
        
        # Start playback
        print(f"Starting playback of: {args.recording_file}")
        print(f"Start time: {args.start_time}s")
        if args.duration > 0:
            print(f"Duration: {args.duration}s")
        else:
            print("Duration: Full recording")
            
        client.replay_file(args.recording_file, args.start_time, args.duration, 37)

        
        print("Playback started successfully!")
        print("Press Ctrl+C to stop playback")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping playback...")
            client.stop_replayer()
            print("Playback stopped.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()