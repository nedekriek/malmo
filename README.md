malmo

The basic protocol:
    If you are working on a new feature make a new branch. When that branch is done and bug free merge it into the master. Git will let you know if your changes conflict with someone elses work git will let us know and we can resolve it. When you make a commit please always include a short message of the latest updates/changes to the branch. 

Making and Merging branches:
    https://stackabuse.com/git-merge-branch-into-master/
    Note: if you want to see what branches exist use git branch

How to make a commit:
    0. be sure you are in the correct branch
    1. git add {filename}   
        This will stage just the file you specify      
       OR         
    1. git add .  
        This will stage all files git detects a change in 
    2. git commit -m "{your commit message}"
        This will commit your staged changes to your local local repository 
        2.1 if you want to add anything that you forgot to the current commit you can stage it the use the following:
            git commit --amend 
    3. git push 
        This sends your commit to the remote location. When this complete others can now access the code from the remote in the remote representation of the branch you committed it from.

How to change branches:
    git checkout {branch name}