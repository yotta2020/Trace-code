def get_task_type(opt):
    if opt.task in ["defect", "clone"]:
        return "classify"
    elif opt.task in ["codesearch", "summarize", "translate", "refine"]:
        return "generate"
    else:
        return "generate"
