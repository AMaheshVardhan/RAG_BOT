import uuid

class MCPMessage:
    def __init__(self, sender, receiver, type_, payload, trace_id=None):
        self.sender = sender
        self.receiver = receiver
        self.type = type_
        self.trace_id = trace_id or str(uuid.uuid4())
        self.payload = payload

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "trace_id": self.trace_id,
            "payload": self.payload
        }

    def __repr__(self):
        return str(self.to_dict())