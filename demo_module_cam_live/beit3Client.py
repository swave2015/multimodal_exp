import grpc
from grpc_protos import beit3service_pb2
from grpc_protos import beit3service_pb2_grpc
import io
import torch
import time
import cv2

class GRPCClient:

    def __init__(self):
        self.channel = grpc.insecure_channel('192.168.31.28:3000')
        self.stub = beit3service_pb2_grpc.Beit3ServiceStub(self.channel)

    def send_tensor_to_server(self, image, frame_id, tracker_id):
        # Serialize the tensor
        # tensor_buffer = io.BytesIO()
        # torch.save(tensor, tensor_buffer)
        # serialized_tensor = tensor_buffer.getvalue()

        # Create the request
        request = beit3service_pb2.Beit3Request(
            image=image,
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
    image = cv2.imread('/home/caoxh/multimodal_exp/test_imgs/frame0337.jpg')
    image = cv2.resize(image, (384, 384))
    print('input_img_shape: ', image.shape)
    # ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    ret, buffer = cv2.imencode('.jpg', image)
    image = buffer.tobytes()
    # tensor = torch.rand(3, 384, 384)
    # tensor = torch.randint(10, size=(3, 384, 384)) 
    frame_id = "YOUR_FRAME_ID"
    tracker_id = "YOUR_TRACKER_ID"

    elapsed_times = []
    num_calls = 100

    for _ in range(num_calls):
        retcode, elapsed = client.send_tensor_to_server(image, frame_id, tracker_id)
        elapsed_times.append(elapsed)

    avg_time = sum(elapsed_times) / len(elapsed_times)
    max_time = max(elapsed_times)
    min_time = min(elapsed_times)

    print(f"Average time for gRPC call: {avg_time:.2f} ms")
    print(f"Max time for gRPC call: {max_time:.2f} ms")
    print(f"Min time for gRPC call: {min_time:.2f} ms")
