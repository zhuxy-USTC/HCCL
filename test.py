import git
import subprocess

repo_dir = '/tmp/cann-hccl'
commit_message = 'Automated commit via script'
token = 'b490f00ee9f73e011bfa33fb28e20054'
username = 'baigj'

# GitHub: https://<token>@github.com/username/repo.git
# Gitee:  https://<token>@gitee.com/username/repo.git
remote_url = f'https://{username}:{token}@gitee.com/baigj/cann-hccl.git'


# subprocess.run(['cd', '/tmp'])
# subprocess.run(['git', 'clone', remote_url])

# subprocess.run(['cd', 'cann-hccl'])
 
# subprocess.run(['echo', '"11111111111"', '>', 'a.txt'])

repo = git.Repo(repo_dir)

repo.git.add(A=True)


repo.index.commit(commit_message)

origin = repo.remote(name='origin')
origin.set_url(remote_url)

origin.push()

print("Changes pushed to the remote repository successfully.")

