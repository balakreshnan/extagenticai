# Install python

- go to extensions
- search python
- install python and python debugger
- now to to view -> command palatte -> python: Create environment
- Select python 3.12 version
- use the requirements.txt file to install all packages

## install azd 

- here is the document to install azd - https://github.com/MicrosoftDocs/azure-dev-docs/blob/main/articles/azure-developer-cli/install-azd.md
- without login scripts wont run, as it can't authenticate.

```
curl -fsSL https://aka.ms/install-azd.sh | bash
```

- in case to understand uninstall

```
curl -fsSL https://aka.ms/uninstall-azd.sh | bash
```

- Now run azd auth login

```
azd auth login
```

- Copy the code the login with browser