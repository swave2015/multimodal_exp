# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import beit3service_pb2 as beit3service__pb2


class Beit3ServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.EnqueueItem = channel.unary_unary(
                '/beit3service.Beit3Service/EnqueueItem',
                request_serializer=beit3service__pb2.Beit3Request.SerializeToString,
                response_deserializer=beit3service__pb2.Beit3Response.FromString,
                )


class Beit3ServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def EnqueueItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_Beit3ServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'EnqueueItem': grpc.unary_unary_rpc_method_handler(
                    servicer.EnqueueItem,
                    request_deserializer=beit3service__pb2.Beit3Request.FromString,
                    response_serializer=beit3service__pb2.Beit3Response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'beit3service.Beit3Service', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Beit3Service(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def EnqueueItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/beit3service.Beit3Service/EnqueueItem',
            beit3service__pb2.Beit3Request.SerializeToString,
            beit3service__pb2.Beit3Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
