class File:
    def __init__(self, name, size):
        self.name = str(name)
        self.size = float(size)
        self.real_input = True  # not produced by a task
        self.producer_task_id = None
        self.consumer_tasks_id = []

    def __str__(self):
        return "File (name: {}, size: {}, consumers: {})".format(
            self.name, self.size, self.consumer_tasks_id
        )

    def __repr__(self):
        return self.name
