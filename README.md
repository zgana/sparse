# sparse

This is an abandoned python sparse matrix module with faster construction but
slower arithmetic than scipy.sparse.  Basically, I profiled some code and found
that sparse matrix construction was a bottleneck.  So I tried writing my own.
Then I profiled my code again, and sparse matrix construction was no longer a
bottleneck.  Instead, sparse matrix dot products was the bottleneck.

So, lol, so much for that.

This is just uploaded in case I ever want to revisit it, either to make it
actually good, or just to remind myself why it isn't good.

## license


            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                        Version 2, December 2004 

     Copyright (C) 2004 Sam Hocevar <sam@hocevar.net> 

     Everyone is permitted to copy and distribute verbatim or modified 
     copies of this license document, and changing it is allowed as long 
     as the name is changed. 

                DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
       TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

      0. You just DO WHAT THE FUCK YOU WANT TO.

