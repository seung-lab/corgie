Distributed Computation
=======================

``corgie`` breaks large processing jobs into smaller tasks that can fit in memory.
By default, ``corgie`` will execute each task locally in the porcess that invoked the command.
User can supply ``--queue_name`` command argument, in which case tasks will be pushed to the specified   
queue instead of being executed locally. Currently ``corgie`` supports either using existing AWS SQS 
queue name or `FileQueue <https://github.com/seung-lab/python-task-queue#notes-on-file-queue>`_ path for 
the ``--queue_name`` value.

In order to execute tasks from the given queue, run the following command::

   corgie-worker --queue_name {queue name} --lease_seconds {lease seconds}
