---


---

<p><img src="https://kaggle2.blob.core.windows.net/datasets-images/18/18/default-backgrounds/dataset-cover.jpg" alt="enter image description here"></p>
<h1 id="naive-bayes-on-amazon-fine-food-reviews">Naive Bayes on Amazon Fine Food Reviews</h1>
<p>Analyze ~500,000 food reviews from Amazon</p>
<blockquote>
<p><strong>About this Dataset</strong></p>
</blockquote>
<h3 id="context">Context</h3>
<p>This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories</p>
<h3 id="contents">Contents</h3>
<ul>
<li>
<p>Reviews.csv: Pulled from the corresponding SQLite table named Reviews in database.sqlite</p>
</li>
<li>
<p>database.sqlite: Contains the table ‘Reviews’</p>
</li>
</ul>
<p>Data includes:</p>
<ul>
<li>Reviews from Oct 1999 - Oct 2012</li>
<li>568,454 reviews</li>
<li>256,059 users</li>
<li>74,258 products</li>
<li>260 users with &gt; 50 reviews</li>
</ul>
<blockquote>
<h3 id="table-of-contents">Table of Contents</h3>
</blockquote>
<ol>
<li>Loading the dataset</li>
<li>Pre-processing the dataset</li>
<li>Feature engineering
<ul>
<li>Review Text --&gt;Text Vector</li>
</ul>
</li>
<li>Naive Bayes with different hyperparameter</li>
<li>Accuracy</li>
</ol>
<blockquote>
<h3 id="output-sample">Output Sample</h3>
</blockquote>
<pre><code> Generalization accuracy = 0.8648484848484849
             precision    recall  f1-score   support

   negative       0.42      0.48      0.45       376
   positive       0.93      0.91      0.92      2924

avg / total       0.87      0.86      0.87      3300
</code></pre>
<hr>
<blockquote>
<p><strong>Naive Bayes Intuition</strong></p>
</blockquote>
<ul>
<li>credit --&gt; <a href="https://www.geeksforgeeks.org/naive-bayes-classifiers/">geeksforgeeks.org</a></li>
<li><strong>Bayes’ Theorem</strong></li>
</ul>
<p>Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7777aa719ea14857115695676adc0914_l3.svg" alt=" P(A|B) = \frac{P(B|A) P(A)}{P(B)} " title="Rendered by QuickLaTeX.com"></p>
<p>where A and B are events and P(B) ? 0.</p>
<ul>
<li>Basically, we are trying to find probability of event A, given the event B is true. Event B is also termed as  <strong>evidence</strong>.</li>
<li>P(A) is the  <strong>priori</strong>  of A (the prior probability, i.e. Probability of event before evidence is seen). The evidence is an attribute value of an unknown instance(here, it is event B).</li>
<li>P(A|B) is a posteriori probability of B, i.e. probability of event after evidence is seen.</li>
</ul>
<p>Now, with regards to our dataset, we can apply Bayes’ theorem in following way:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e85875a7ff9e9b557eab6281cc7ff078_l3.svg" alt=" P(y|X) = \frac{P(X|y) P(y)}{P(X)} " title="Rendered by QuickLaTeX.com"></p>
<p>where, y is class variable and X is a dependent feature vector (of size  <em>n</em>) where:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5385a4693c3fb17811cf36593978a601_l3.svg" alt=" X = (x_1,x_2,x_3,.....,x_n) " title="Rendered by QuickLaTeX.com"></p>
<p><strong>Naive assumption</strong></p>
<p>Now, its time to put a naive assumption to the Bayes’ theorem, which is,  <strong>independence</strong>  among the features. So now, we split  <strong>evidence</strong>  into the independent parts.</p>
<p>Now, if any two events A and B are independent, then,</p>
<pre><code>P(A,B) = P(A)P(B)

</code></pre>
<p>Hence, we reach to the result:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1c3f5ab570cf0ab3f43d5c18c645b67a_l3.svg" alt=" P(y|x_1,...,x_n) = \frac{ P(x_1|y)P(x_2|y)...P(x_n|y)P(y)}{P(x_1)P(x_2)...P(x_n)} " title="Rendered by QuickLaTeX.com"></p>
<p>which can be expressed as:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8171c1fe2cbd3ed62bc3f40d682c0512_l3.svg" alt=" P(y|x_1,...,x_n) = \frac{P(y)\prod_{i=1}^{n}P(x_i|y)}{P(x_1)P(x_2)...P(x_n)} " title="Rendered by QuickLaTeX.com"></p>
<p>Now, as the denominator remains constant for a given input, we can remove that term:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c778553cb5a67518205ac6ea18502398_l3.svg" alt=" P(y|x_1,...,x_n)\propto P(y)\prod_{i=1}^{n}P(x_i|y) " title="Rendered by QuickLaTeX.com"></p>
<p>Now, we need to create a classifier model. For this, we find the probability of given set of inputs for all possible values of the class variable  <em>y</em>  and pick up the output with maximum probability. This can be expressed mathematically as:</p>
<p><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f3637f468262bfbb4accb97da8110028_l3.svg" alt="y = argmax_{y} P(y)\prod_{i=1}^{n}P(x_i|y) " title="Rendered by QuickLaTeX.com"></p>
<p>So, finally, we are left with the task of calculating P(y) and P(xi  | y).</p>
<p>Please note that P(y) is also called  <strong>class probability</strong>  and P(xi  | y) is called  <strong>conditional probability</strong>.</p>

