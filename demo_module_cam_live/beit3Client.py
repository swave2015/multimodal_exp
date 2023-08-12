import grpc
from grpc_protos import beit3service_pb2
from grpc_protos import beit3service_pb2_grpc
import io
import torch
import time

class GRPCClient:

    def __init__(self):
        self.channel = grpc.insecure_channel('192.168.31.28:3000')
        self.stub = beit3service_pb2_grpc.Beit3ServiceStub(self.channel)

    def send_tensor_to_server(self, tensor, frame_id, tracker_id):
        # Serialize the tensor
        tensor_buffer = io.BytesIO()
        torch.save(tensor, tensor_buffer)
        serialized_tensor = tensor_buffer.getvalue()

        # Create the request
        request = beit3service_pb2.Beit3Request(
            serialized_tensor=serialized_tensor,
            frame_id=frame_id,
            tracker_id=tracker_id
        )

        # Start timing
        start_time = time.time()

        # Send the request
        response = self.stub.EnqueueItem(request)

        # End timing
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"elapsed_time: {elapsed_time}")

        return response.retcode, elapsed_time

if __name__ == "__main__":
    client = GRPCClient()
    
    # tensor = torch.rand(3, 10, 10)
    tensor = torch.randint(10, size=(3, 384, 384)) 
    frame_id = "YOUR_FRAME_ID"
    tracker_id = "YOUR_TRACKER_ID"

    elapsed_times = []
    num_calls = 100

    for _ in range(num_calls):
        retcode, elapsed = client.send_tensor_to_server(tensor, frame_id, tracker_id)
        elapsed_times.append(elapsed)

    avg_time = sum(elapsed_times) / len(elapsed_times)
    max_time = max(elapsed_times)
    min_time = min(elapsed_times)

    print(f"Average time for gRPC call: {avg_time:.2f} ms")
    print(f"Max time for gRPC call: {max_time:.2f} ms")
    print(f"Min time for gRPC call: {min_time:.2f} ms")
