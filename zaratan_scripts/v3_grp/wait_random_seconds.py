import time
import random
import sys

# Get the subject ID (job index)
subject_id = int(sys.argv[1])

print(f'start time {time.time()}')

# Wait for a random number of seconds between 1 and 10
wait_time = random.randint(1, 10)
print(f"Subject {subject_id}: Waiting for {wait_time} seconds...")

# Sleep for the random time
time.sleep(wait_time)

print(f"Subject {subject_id}: Done waiting.")

print(f'end time {time.time()}')