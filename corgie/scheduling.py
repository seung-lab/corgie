import mazepa

wait_until_done = mazepa.Barrier
Task = mazepa.Task
Executor = mazepa.Executor
scheduler_click_options = mazepa.click_options
parse_scheduler_from_kwargs = mazepa.parse_scheduler_from_kwargs
parse_executor_from_kwargs = mazepa.parse_executor_from_kwargs

class Job(mazepa.Job):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def task_generator():
        raise NotImplemented("Job classes must implement "
                "'task_generator' function")

    def get_tasks(self):
        return next(self.task_generator)

class Scheduler(mazepa.Scheduler):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

def create_scheduler(*kargs, **kwargs):
    return Scheduler(*kargs, **kwargs)

