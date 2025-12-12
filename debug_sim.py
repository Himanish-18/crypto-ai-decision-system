
from src.sim.event_bus import sim_bus

def handler(payload):
    print(f"Time: {sim_bus.current_time} | Payload: {payload}")
    if sim_bus.current_time < 2.0:
        sim_bus.publish("ping", payload + 1, delay=1.0)

sim_bus.subscribe("ping", handler)
sim_bus.publish("ping", 1, delay=0.1)

print("Starting Loop...")
count = sim_bus.run_until(5.0)
print(f"Finished. Processed {count} events.")
