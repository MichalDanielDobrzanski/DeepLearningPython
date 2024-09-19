[33mcommit 2da41b6dd19dafae92db1facbc4ed19738e80551[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmaster[m[33m, [m[1;31morigin/master[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: michaldobrzanski <dobrzanski.michal.daniel@gmail.com>
Date:   Mon Nov 27 14:07:08 2023 +0100

    added MIT license

[33mcommit 2eae26e0bdcef314dcb18f13946a94320fb28a12[m
Merge: ea229ac 11cd4c1
Author: Michał Dobrzanski <dobrzanski.michal.daniel@gmail.com>
Date:   Wed Feb 24 08:57:56 2021 +0100

    Merge pull request #32 from hamolicious/patch-1
    
    Update network.py

[33mcommit 11cd4c1643c39ada6a97b93dc2c522a3b59095ac[m
Author: Hamolicious <56944714+hamolicious@users.noreply.github.com>
Date:   Fri Oct 23 13:12:15 2020 +0100

    Update network.py
    
     - Removed semicolon line:73

[33mcommit ea229ac6234b7f3373f351f0b18616ca47edb8a1[m
Merge: a8da42f 998e51b
Author: Michał Dobrzanski <dobrzanski.michal.daniel@gmail.com>
Date:   Thu Apr 19 07:36:33 2018 +0200

    Merge pull request #11 from dgcampbe/patch-1
    
    Small update for Python 3

[33mcommit 998e51b741e95c2e6fd18cdba7df0c9b94e7957f[m
Author: Dane <dgcampbe@users.noreply.github.com>
Date:   Fri Mar 2 17:04:26 2018 -0500

    Add str() to variables being printed
    
    I realized that the variables weren't strings despite having "%s" (I thought the "s" meant string) in Python 2. That is what you get for only dealing with Python 3. :P

[33mcommit ce2c044fb2b85371d65b4c71ab08b3bed9ba8067[m
Author: Dane <dgcampbe@users.noreply.github.com>
Date:   Fri Mar 2 01:42:31 2018 -0500

    Small update for Python 3
    
    Changed two print statements to print functions to be compatible with Python 3 instead of Python 2.

[33mcommit a8da42fef60e9de977e3ad382c0804a1fe58cbc5[m
Merge: f48defe 2a26d29
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Mon Aug 28 11:01:48 2017 +0200

    Merge pull request #2 from pa-m/master
    
    Avoid ImportError: cannot import name 'downsample'. New naming convention has been added in Theano 0.9 => use pool_2d instead

[33mcommit 2a26d2937c2412c5ece5d26587a7351dcb226b67[m
Author: pa-m <pascal.masschelier@gmail.com>
Date:   Tue Apr 4 19:39:27 2017 +0200

    Avoid ImportError: cannot import name 'downsample'
    Since a change in theano 0.9 downsample.max_pool_2d has to be replaced with pool.pool_2d.

[33mcommit f48defe06bfa1127fbda4f054b27d1e23c17c725[m
Merge: e99416e 3ef0f0b
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Thu Feb 9 10:31:15 2017 +0100

    Merge pull request #1 from anandman/master
    
    updated mnist_average_darkness.py for Python 3.5.2 - further improvements

[33mcommit 3ef0f0b575e0a6a96508fbb9ae6c51be2def8672[m
Author: anand m <anand@resistance.net>
Date:   Sat Feb 4 13:58:16 2017 -0800

    updated for Python 3.5.2

[33mcommit e99416e6ad59e602d0dc3709a445c3a4efad259f[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Sun Oct 23 12:00:59 2016 +0200

    Update README.md

[33mcommit a35f06c85e2903c81e3f9394ad7e5f24443e40ae[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Sun Oct 23 11:57:30 2016 +0200

    Working on network3.py - convolutional NN examples added

[33mcommit 664aee31f815956aaef96a533521d1eddd5fb283[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Sun Sep 25 16:56:45 2016 +0200

    Working Theano with CUDA7.5 Support at Ubuntu 16.04

[33mcommit f403c099829ee70700b60569a546ab7e9d6793b9[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Wed Sep 14 19:46:01 2016 +0200

    Working on network3.py and Theano with CUDA

[33mcommit 092007f9158a4a3ff56a08d03250a19ae8953753[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Tue Sep 13 11:51:49 2016 +0200

    Chapter 4 - vanishing gradient problem in a deep networks

[33mcommit 7283525363f379b99779c96b6cc82b25626c2249[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Fri Sep 9 15:28:59 2016 +0200

    Early stopping implemented at network2.py

[33mcommit 021657efa2f874453ebf94ca1598b38add9b7dc5[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Sat Jul 30 18:08:54 2016 +0200

    Updated readme + 2nd example from 3rd chapter

[33mcommit 988965a43e7b002c5bc03ecb675e9c0f15ba9072[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Fri Jul 29 20:23:55 2016 +0200

    Create README.md

[33mcommit 0edab8e2501621752d6dab41bf193a380eb2819b[m
Author: Michał Dobrzański <dobrzanski.michal.daniel@gmail.com>
Date:   Fri Jul 29 20:06:48 2016 +0200

    Initial files
