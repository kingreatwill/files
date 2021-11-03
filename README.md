# files

## 大文件存储
突破GitHub的限制，使用 git-lfs(Git Large File Storage) 支持单个文件超过100M

1. 下载Git Large File Storage (LFS)：https://git-lfs.github.com/
2. `git lfs install`
3. `git lfs track "*.pdf"` and `git lfs track "*.ppt"`
会出现.gitattributes文件
4. 正常的提交

对这些大文件（默认10M）在git的repository里面只存一个小的文本文件，这个文本文件描述了要去哪里下载对应的二进制文件。

[git lfs migrate](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-migrate.1.ronn)