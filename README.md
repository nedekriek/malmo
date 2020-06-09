# Notes on our implementation
Setting up Malmo can be a bit of a pain, I can add rough instructions on how to do this later.

We have a hardcoded dictionary in our "main_run_multiple" and "main_with_ga_nn" files that you can change. The later runs one iteration of the agents using the given parameters, and the former can be used to set up incremental tests or multiple tests at once.

To run this you will need to run a Malmo Minecraft client, and then run the "main_run_multiple" or "main_with_ga_nn" files.

The tests are sped up to 5x speed with a max time of 30 seconds (6 seconds after speed up). Some parameters can be changed in the scenarios.xml file. 


# Notes for us: The basic protocol:

If you are working on a new feature make a new branch. When that branch is done and bug free merge it into the master. Git will let you know if your changes conflict with someone elses work git will let us know and we can resolve it. When you make a commit please always include a short message of the latest updates/changes to the branch. 

## Making and Merging branches:
https://stackabuse.com/git-merge-branch-into-master/

I have been deleting branches after I have merged them into master so far. This is just so everyone starts with a clean slate. You do not have to bother with deleting branches at all. Just make sure you are using good naming practices. 

Note: if you want to see what branches exist use git branch 


## What is the difference between git pull and git rebase?

https://stackoverflow.com/questions/36148602/git-pull-vs-git-rebase

## How to make a commit:
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

## How to change branches:
    git checkout {branch name}

## How to see the different branches in the repo:
    git branch

## How to get code from a remote repo to your local machine for a specific branch:
    git pull {branch name}

## How to look at the commit history on a branch:
    git log

## What is git status:
It displays the state of the working directory and the staging area. It lets you see which changes have been staged, which haven't, and which files aren't being tracked by Git.

## How to look at the status of your local branch:
    git status

## How to look at the status of your local branch compared to the remote branch for a specific branch:
    git status {upstream} {branch name}

## How to see the remotes:
    git remote -v
    How to add upstream
        git add upstream "{link to repo}"