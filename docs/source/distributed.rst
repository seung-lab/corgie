Distributed Computation
=======================

To use distributed computation use ``--queue_name`` argument on the master side command.

Each worker needs to execute::

   corgie-worker --queue_name {queue name} --lease_seconds {lease seconds}
