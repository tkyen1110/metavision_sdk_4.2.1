# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
"JoinThread" class that iterate in a separate thread!
This is based on the code of Pytorch DataLoader
"""
import torch
import threading
import queue
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils.pin_memory import pin_memory
from torch._utils import ExceptionWrapper


MISSING = object()


class JoinThread(object):
    def __init__(self, base_iterator, pin_memory):
        self.base_iterator = base_iterator
        self.pin_memory = pin_memory

    def join_thread(self, out_queue, device_id):
        for data in self.base_iterator:
            if self.pin_memory and not isinstance(data, ExceptionWrapper):
                data = pin_memory(data)
            out_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)
        out_queue.put(MISSING)

    def __iter__(self):
        data_queue = queue.Queue()
        join_memory_thread = threading.Thread(
            target=self.join_thread,
            args=(
                data_queue,
                torch.cuda.current_device(),
            ),
        )
        join_memory_thread.daemon = True
        join_memory_thread.start()
        while True:
            data = data_queue.get(timeout=100000)
            if data is MISSING:
                raise StopIteration
            yield data
