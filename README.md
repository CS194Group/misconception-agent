# Math agent

> This is a group project of cs 194/294 that focusing on finding the misconceptions in the math problems. 

### Installation

> Basically, this repo just need two main dependency, you may fix the import error when you encounter them in specific file.

```
pip install dspy
pip install pydantic
```

### Folder Explainations

#### *agents*

> This contains the old version of agents that we're using. 
>
> 1. BasexxxAgent: represent basic agents that only consist of a single CoT.
> 2. AgentGroup: use BaseAgent to construct larger architecture.
> 3. MathAgent: main function
>
> > [!TIP]
> >
> > - You don't need to modified this folder(and if you want, go ahead). Since our current agents don't use them.
> > - Fell free to use them as reference when you are creating new agents.

#### *data*

> This folder contains all the data we'll use in the final project

#### *datamanager*

> Use DataManager in the template file to load the data.

#### *EoT*

> Original code of the paper EoT
>
> As well as amaanxansari's EoT template.

#### *milestone1*

> The version we're using now.
>
> 1. agent_test: code entrance
> 2. MathAgent: Agents we use.
> 3. rag_test: Test to retrieve data from the ***data***
> 4. utils: Some helper functions
>
> > [!TIP]
> >
> > - If you want to change the architecture. You should change both the **forward** part in the agent_test and add agents you need to use in the **MathAgent**
> > - Currently, the data loading methods is quite different with the one mentioned in the datamanager

#### *other*

> Simple dspy demos that you could test yourself to get familiar with the usage of dspy and pydantic.

#### *test*

> We may put our test code here? I'm not quite sure what this folder do.

# Dev Notes
Please use pytest for writing unit tests!
Please use pathlib.Path to define a path so it works on all operating systems.
