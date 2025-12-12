import zmq
import logging
import json

logger = logging.getLogger("dist_bus")

class EventBus:
    """
    v30 Distributed Event Bus (ZeroMQ).
    Decouples Alpha Engine from Execution Engine.
    """
    def __init__(self, pub_port=5555, sub_port=5556, role="PUB"):
        self.context = zmq.Context()
        self.role = role
        
        if role == "PUB":
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{pub_port}")
        else:
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{pub_port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def publish(self, topic: str, message: dict):
        if self.role != "PUB": return
        self.socket.send_string(f"{topic} {json.dumps(message)}")
        
    def listen(self):
        if self.role != "SUB": return
        while True:
            msg = self.socket.recv_string()
            topic, data = msg.split(" ", 1)
            yield topic, json.loads(data)
