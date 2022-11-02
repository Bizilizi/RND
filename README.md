<center><b>Experiment #1</b></center>
<center>Reproducing plots from the RND paper. E.g. Figure 2, page 5</center> 

**Dataset:**  
MNIST data set (torch vision as source for it)

**Models**  
Two networks: Learnable CNN (LCNN), Fixed CNN (FCNN). Both uses different set of parameters and potentially architectures. 

**Experiment parameters:**
- Number of samples from source class n_s.   
- Number of samples from target class n_t.

**Train setup**  
We train LCNN on dataset consisted from target and source classes until we reach overfitting, e.g. considerably small L2 loss between prediction of FCNN and LCNN.  
Submission is the obtained L2 loss after convergence.

**Expectations**
We expect submitted L2 loss decrease for increasing n_t parameter. 

---
<center><b>Experiment #2</b></center>
<center>Exploring how the L2 loss behaves on fine tunnig with real data</center> 


**Dataset:**  
MNIST data set (torch vision as source for it)

**Models**  
Two networks: Learnable CNN (LCNN), Fixed CNN (FCNN). Both uses different set of parameters and potentially architectures. 

**Experiment parameters:**
- Source class. _This class will be used to learn initial state for LCNN network._  
- Target class. _This class will be used to calculate L2 loss between LCNN and FCNN._

**Pre-Train setup**  
We train LCNN on source class until we reach overfitting, e.g. considerably small L2 loss between prediction of FCNN and LCNN. 

**Post-Train setup**  
We used obtained weight from the pre-train step and add target class to our train dataset. 
At every n epoch of post-Train step, we submit loss between FCNN and LCNN predictions for target class.

**Expectations**
We expect submitted L2 loss in post-Train step decrease with training time.
---
<center><b>Experiment #3</b></center>
<center>Exploring how the L2 loss behaves on fine tunnig with real data</center> 


**Dataset:**  
MNIST data set (torch vision as source for it)

**Sampling**  
Stream the data in per class:, 0’s, then 1’s, then 2’s  
Or, mix classes a bit: lots of 0’s, then a few 1’s, then lots of 2’s

**Models**  
Define a simple CNN
Need to keep the seen points in memory (can just index it, don’t actually need a memory component)

**CL improvement**  
For every new point: Calculate the L2 distance between this point and all the previous seen points.  
If the minimum distance over all of the seen points is below  some threshold.
Sample points (let’s start with just trying out gaussian noise) around this new point (it’s a ‘seen before point’ → We want to not forget it)
Push these points through our simple CNN. This gives you ‘pseudo groundtruths’
And these random points + random pseudo groundtruths to the batch for your gradient calculation.