from clease import settingFromJSON, Evaluate

setting = settingFromJSON("almgsixSettings3.json")
evaluator = Evaluate(setting)
evaluator.export_dataset("data/almgsi_binary_linear.csv")

