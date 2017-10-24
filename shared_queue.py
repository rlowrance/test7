'fascade to enable testing without starting RabbitMQ'
# The classes FileBlockChannel and FileBlockingConnection are stubs in the current implementation.
# The classes are intended to fully emulate a subset of the RabbitMQ system, but they are complex
# and haven't yet been written, so they are stubs at present.
#
# The classes PrimitiveChannel and PrimitiveConnection are implemented. They provide just barely
# enough functionality to get started on development and testing. These classes DO NOT fully
# emulate that RabbitMQ Channel and Connection classes. That means that code developed with them
# may NOT work in the production system. (However, simple use cases will work. See events_cusip.py
# in the machine learning repo for an example of how to use them.)
#
# In the primitive subset:
# - Files are used to hold the queues
# - The caller provides a function that is used by the publish method to
#   map an exchange name (a str) and a routing key (a str) to a file path.
#   Messages are simply appended to the file
# - The caller provides a function that is used by the subscribe method
#   to map a queue name (a str) to a file path. Messages are simply read
#   from the file.
# - All transactions are blocking (nothing is asynchronous)
# - Parameter values for method calls are restricted

import abc
import copy
import pika
import pdb
import time
import typing
import unittest

import shared_message


class PrimitiveBlockingChannel:
    'provide a subset of the functionality of a BlockChannel'
    # ref: http://pika.readthedocs.io/en/latest/modules/adapters/blocking.html
    def __init__(self, path_for_input, paths_for_output):
        self._path_for_input = path_for_input
        self._paths_for_output = paths_for_output
        
        self._consumer_files = {}  # key = path, value = open file
        self._producer_files = {}  # key = path, value = open file

    def __str__(self):
        return 'PrimitiveBlockingChannel(|consumer_files|=%d, |producer_files|=%d)' % (
            len(self._consumer_files),
            len(self._producer_files),
            )

    def close(self,
              reply_code=200,
              reply_text='Normal shutdown',
              ):
        'close the channel by closing all of the files'
        assert reply_code == 200
        assert reply_text == 'Normal shutdown'
        for path, f in self._consumer_files.items():
            f.close()
        for path, f in self._producer_files.items():
            f.close()
        
    def consume(self,
                queue: str,
                no_ack=False,
                exclusive=False,
                arguments=None,
                inactivity_timeout=None,
                ):
        'iterate over message in the file representing the queue, raising StopIteration at end of file'
        if queue not in self._consumer_files:
            path = self._path_for_input(queue)
            f = open(path, 'r')
            self._consumer_files[queue] = f
        line = self._consumer_files[queue].readline()
        if len(line) == 0:
            raise StopIteration  # the file is at end of file
        return line[:-1]  # remove final \n

    def publish(self,
                exchange: str,
                routing_key: str,
                body: str,
                properties=None,
                mandatory=False,
                immediate=False,
                ):
        'append message to file'
        # make sure optional parameters are at default values
        assert isinstance(body, str)  # body is a json-encoded str representing a Message
        assert properties is None
        assert mandatory is False
        assert immediate is False

        paths = self._paths_for_output(exchange, routing_key)
        for path in paths:
            if path not in self._producer_files:
                f = open(path, 'w')
                self._producer_files[path] = f
            f = self._producer_files[path]
            f.write(body)
            f.write('\n')
                      

class PrimitiveBlockingConnection:
    def __init__(self, path_for_input, paths_for_output):
        self._path_for_input = path_for_input
        self._paths_for_output = paths_for_output
        self._channels = []

    def __str__(self):
        return 'PrimitiveBlockingConnection(|channels]=%d)' % (
            len(self._channels),
            )

    def channel(self, channel_number=None):
        'create a new channel on next available channel number'
        assert channel_number is None
        c = PrimitiveBlockingChannel(
            path_for_input=self._path_for_input,
            paths_for_output=self._paths_for_output,
            )
        self._channels.append(c)
        return c

    def close(self,
              reply_code=200,
              reply_text='Normal shutdown',
              ):
        'disconnect from RabbitMQ'
        assert reply_code == 200
        assert reply_text == 'Normal shutdown'
        for channel in self._channels:
            channel.close(
                reply_code=reply_code,
                reply_text=reply_text,
                )


class FileBlockingChannel(pika.adapters.blocking_connection.BlockingChannel):
    'emulate RabbitMQ blocking connections using the file system'
    # ref: http://pika.readthedocs.io/en/0.11.0/modules/channel.html
    def __init__(self, connection, channel_number, on_open_callback):
        'create new channel (the primary means of interacting with RabbitMQ)'
        # create instances by calling FileBlockingConnection.channel()
        assert on_open_callback is None  # not yet implemented (sorry)
        self._connection = connections
        self._channel_number = channel_number
        self._on_open_callback = on_open_callback

    def add_on_cancel_callback(self, callbacke):
        'specify callback function called with basic_cancel sent by server'
        # basic_cancel is never sent by the file system emulation of RabbitMQ
        raise NotImplemented()

    def add_on_close_callback(self, callback):
        'specify callback function called when the channel is closed'
        raise NotImplemented()

    def add_on_flow_callback(self, callback):
        'deprecated method in the pika API'
        raise NotImplemented()

    def add_on_return_callback(self, callback):
        'specify callback function called with a message is rejected'
        raise NotImplemented()

    def basic_ack(self, delivery_tag=0, multiple=False):
        'acknowledge one or more messages'
        # parameters
        #  delivery_tag: assigned by the server when message was delivered (an int or long)
        #  multiple: if True, the delivery tag is treated as an up-to-and-including value
        pass

    def basic_cancel(self, callback=None, consumer_tag: str='', nowait=False):
        'tell server to not send any more message to a consumer'
        raise NotImplemented()

    def basic_consume(self,
                      consumer_callback,
                      queue: str,
                      no_ack=False,
                      exclusive=False,
                      consumer_tag: str=None,
                      arguments=None,
                      ) -> str:
        'specify callback function called for each message on specified queue'
        # TODO: implement me
        raise NotImplemented()
        
    def basid_get(self,
                  callback,
                  queue,
                  no_ack,
    ):
        'get a single message'
        raise NotImplemented()

    def basic_nack(self, delivery_tag=None, multiple=False, requeue=True):
        'reject one or more messages'
        raise NotImplemented()

    def basic_publish(self,
                      exchange: str,
                      routing_key: str,
                      body: str,
                      properties,
                      mandatory: bool,
                      immediate: bool,
                      ):
        'publish to the channel'
        # TODO: implement me
        raise NotImplemented()

    def basic_qos(self,
                  callback=None,
                  prefetch_size: int=0,
                  prefetch_count: int=0,
                  all_channels=False,
                  ):
        'specify qualify of service'
        pass  # fully implemented, but we do nothing regarding QoS

    def basic_reject(self, delivery_tag, requeue=True):
        'reject an incoming message'
        raise NotImplemented()

    def basic_recover(self, callback=None, requeue=False):
        'ask the server to redeliver all unacknowledged messages'
        raise NotImplemented()

    def close(self, reply_code=0, reply_text='Normal shutdown'):
        'invoke a graceful showdown of the channel'
        # parameters
        #  reply_code: reason to send to the broker
        #  reply_text: reason text to send to the broker
        for file in self._files:
            file.close()

    def confirm_delivery(self, callback=None, nowait=False):
        'turn on Confirm mode'
        raise NotImplemented()

    def consumer_tags(self):
        'return list of currently active consumers'
        raise NotImplemented()

    def exchange_bind(self,
                      callback=None,
                      destination=None,
                      source=None,
                      routing_key='',
                      nowait=False,
                      arguments=None,
                      ):
        'bind an exchange to another exchange'
        raise NotImplemented()

    def exchange_declare(self,
                         callback=None,
                         exchange=None,
                         exchange_type='direct',
                         passive=False,
                         durable=False,
                         auto_detect=False,
                         internal=False,
                         nowait=False,
                         arguments=None,
                         ):
        'create an exchange if it does not already exist'
        raise NotImplemented()

    def exchange_delete(self,
                        callback=None,
                        exchange=None,
                        if_unused=False,
                        nowait=False,
                        ):
        'delete the exchange'
        raise NotImplemented()

    def exchange_unbind(self,
                        callback=None,
                        destination=None,
                        source=None,
                        routinte_key='',
                        nowait=False,
                        arguments=None,
                        ):
        'unbind an exchange from another exchange'
        raise NotImplemented()

    def flow(self, callback, active):
        'turn channel flow control off and on'
        pass  # flow control doesn't exist in the file system implementation

    def is_closed(self):
        'return True iff the channel is closed'
        return not self._is_open

    def is_closing(self):
        'return True if client-initiated closing of the channel is in progress'
        return False

    def is_open(self):
        'return True iff the channel is open'
        return self._is_open

    def open():
        'open the channel'
        # lazily open files when consume and publish method calls are executed'
        self._is_open = True

    def queue_bind(self,
                   callback,
                   queue,
                   exchange,
                   routing_key=None,
                   nowait=False,
                   arguments=None,
                   ):
        'bind the queue to the specified exchange'
        raise NotImplemented()

    def queue_declare(self,
                      callback,
                      queue='',
                      passive=False,
                      durable=False,
                      exclusive=False,
                      auto_delete=False,
                      nowait=False,
                      arguments=None,
                      ):
        'declare queue, creating it if needed'
        raise NotImplemented()

    def queue_delete(self,
                     callback=None,
                     queue='',
                     if_unused=False,
                     if_empty=False,
                     nowait=False,
                     ):
        'delete a queue from the broker'
        raise NotImplemented()

    def queue_purge(self,
                    callback=None,
                    queue='',
                    nowait=False,
                    ):
        'purge all the message from the specified queue'
        raise NotImplemented()

    def queue_unbind(self,
                     callback=None,
                     queue='',
                     exchange=None,
                     routing_key=None,
                     arguments=None,
                     ):
        'unbind a queue fron an exchange'
        raise NotImplemented()

    def tx_commit(self,
                  callback=None,
                  ):
        'commit a transaction'
        raise NotImplemented()

    def tx_rollback(self,
                    callback=None,
                    ):
        'rollback a transaction'
        raise NotImplemented()

    def tx_select(self,
                  callback=None,
                  ):
        'select standard transaction mode'
        raise NotImplemented()

    def start_consuming():
        'repeatedly read message and call the callback function specified by basic_consumer'
        # raise exception to exit; so caller should use a try-except blocking
        # TODO: implement me
        raise NotImplemented()

    def stop_consuming():
        pass


class FileBlockingConnection(pika.adapters.blocking_connection.BlockingConnection):
    'emulate RabbitMQ BlockingConnect using the file system'
    # ref: http://pika.readthedocs.io/en/0.11.0/modules/adapters/blocking.html
    # implement all of the methods
    def __init__(self, parameters=None, _impl_class=None):
        assert parameters is None
        assert _impl_class is None
        self._last_channel_number = 0
        self._channels = []  # typing.List[FileBlockingChannel]
        pass

    def add_on_connection_unblocked_callback(self, callback_method):
        raise NotImplemented()

    def add_timeout(self, deadline, callback_method):
        raise NotImplemented()

    def basic_nack(self):
        pass

    def basic_nack_supported(self):
        return True

    def channel(self, channel_number=None) -> FileBlockingChannel:
        'create new channel with next available channel number'
        assert channel_number is None
        self._last_channel_number += 1
        fbc = FileBlockingChannel(
            connection=self,
            channgel_number=self._last_channel_number,
            on_open_callback=None,
        )
        self._channels.append(fbc)
        return fbc

    def close(self, reply_code=200, reply_text='Normal shutdown'):
        'disconnect from RabbitMQ, closing any open channels'
        for channel in self._channels:
            channel.close()

    def consumer_cancel_notify(self):
        raise NotImplemented()

    def consumer_cancel_notify_supported(self):
        raise NotImplemented()

    def exchange_exchange_bindings(self):
        raise NotImplemented()

    def exchange_exchange_bindings_supported(self):
        raise NotImplemented()

    def is_closed(self):
        'return boolean reporting current connection state'
        return False

    def is_closing(self):
        'return True iff connection is in process of closing'
        return False

    def is_open(self):
        'return boolean reporting current connection state'
        return True

    def process_data_events(self, time_limit=2):
        raise NotImplemented()

    def publisher_confirms(self):
        'specifies if the active connection can use publisher confirmations'
        return True

    def publisher_confirms_supported(self):
        'specifies if the active connection can use publisher confirmation'
        return True

    def remove_timeout(self, timeout_id):
        raise NotImplemented()
        
    def sleep(self, duration):
        time.sleep(duration)


###########################################################
class Test(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid pyflake warnings
        pdb
