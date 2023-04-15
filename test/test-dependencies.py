from maelzel.dependencies import checkDependencies

print("Checking dependencies")
print(checkDependencies(abortIfErrors=False, tryfix=True))

