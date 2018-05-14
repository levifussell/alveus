# Place all code for experiments here

To import from the alveus part of the project include this in the imports section of 
your file.

```
from sys import path
path.insert(0, '../alveus/')
```

Since Alveus is not currently a library installable by pip you have to add the alveus 
the location of the alveus directory to the system path. This tells python where
to search for the alveus modules. You can then do imports of alveus items as you would
if the experiments folder were a part of the alveus folder/module.

For more information see 
[python modules documentation](https://docs.python.org/3/tutorial/modules.html)
and 
[stackoverflow post](https://stackoverflow.com/questions/4383571/importing-files-from-different-folder)
