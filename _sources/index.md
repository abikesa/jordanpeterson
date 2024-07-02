


<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    processEscapes: true
  }
});
</script>


<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        .video-container {
            display: flex;
            justify-content: space-around; /* Adjust the spacing between iframes */
            gap: 20px; /* Add a gap between iframes */
            padding: 20px; /* Add padding around the container */
        }
        .video-container iframe {
            width: 30%; /* Adjust the width of each iframe */
            height: 300px; /* Adjust the height as needed */
            border: none; /* Remove the border */
        }
        body {
            margin: 0; /* Remove margin from body */
            padding: 0; /* Remove padding from body */
            background-color: #ffffff; /* Ensure the background is white */
        }
    </style>
    <title>Re::Melody, Chords, Rhythm</title>
</head>
<body> 
    <div class="video-container">
        <iframe src="https://www.youtube.com/embed/B51na4gPbQU" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <iframe src="https://www.youtube.com/embed/Lu3T0f-H3JI" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <iframe src="https://www.youtube.com/embed/dmZpDa2FLYQ" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
</body>
</html>

# Brahms

> *[Letters to Abigail](https://www.masshist.org/digitaladams/archive/doc?id=L17800512jasecond)*

- Activation Function, $Q$: 1st, 3rd, 5th, [♭♭7th](https://en.wikipedia.org/wiki/Chord_notation#Chord_quality), 9th  
   - [Hunter](https://en.wikipedia.org/wiki/Luis_Palau)-[gatherer](https://en.wikipedia.org/wiki/Eurofest_%2775)/`War`: spiritual teachings  (I)
   - Peasant/`Economics`: [humanism](https://www.uuftc.org) (B)
        
-  [Biases](https://www.youtube.com/watch?v=lAcYahc74o8), $U()$: 11th, 13th
   - [Farmer](https://en.wikipedia.org/wiki/Explo_%2772)/`Calculus`: judeo, christian (G)
   - [Manufacturer](https://www.latimes.com/archives/la-xpm-1986-01-04-me-24254-story.html)/`Philosophy`: world religions (Y)
       
- Weights, $\frac{dU()}{dQ}$: ♯9,♭9,♯11,♭13 
   - Electricity/`Musick`: prophetic utterances ([O](https://www.youtube.com/watch?v=1aM1KYvl4Dw))
   - Railway/`Leisure`: [individual](gpt4o.md) experience ([R](https://www.youtube.com/watch?v=fu-3WN9TJNI))     


```python
import numpy as np
import matplotlib.pyplot as plt

# Define the total utility function U(Q)
def total_utility(Q):
    return 100 * np.log(Q + 1)  # Logarithmic utility function for illustration

# Define the marginal utility function MU(Q)
def marginal_utility(Q):
    return 100 / (Q + 1)  # Derivative of the total utility function

# Generate data
Q = np.linspace(1, 100, 500)  # Quantity range from 1 to 100
U = total_utility(Q)
MU = marginal_utility(Q)

# Plotting
plt.figure(figsize=(14, 7))

# Plot Total Utility
plt.subplot(1, 2, 1)
plt.plot(Q, U, label=r'Total Utility $U(Q) = 100 \log(Q + 1)$', color='blue')
plt.title('Total Utility')
plt.xlabel('Quantity (Q)')
plt.ylabel('Total Utility (U)')
plt.legend()
plt.grid(True)

# Plot Marginal Utility
plt.subplot(1, 2, 2)
plt.plot(Q, MU, label=r'Marginal Utility $MU(Q) = \frac{dU(Q)}{dQ} = \frac{100}{Q + 1}$', color='red')
plt.title('Marginal Utility')
plt.xlabel('Quantity (Q)')
plt.ylabel('Marginal Utility (MU)')
plt.legend()
plt.grid(True)

# Adding some calculus notation and Greek symbols
plt.figtext(0.5, 0.02, r"$MU(Q) = \frac{dU(Q)}{dQ} = \lim_{\Delta Q \to 0} \frac{U(Q + \Delta Q) - U(Q)}{\Delta Q}$", ha="center", fontsize=12)

plt.tight_layout()
plt.show()
```

Running this code will generate a visual demonstration of diminishing marginal utility with appropriate calculus notation and Greek symbols.

Here is the generated image:

![Diminishing Marginal Utility](https://abikesa.github.io/johnadams/diminishing_marginalutility.png)

> One needs [challenges](https://www.voanews.com/a/apple-defying-the-times-stays-quiet-on-ai-/7128857.html), a [worthy adversary](https://www.quora.com/Why-isnt-Apple-part-of-the-Partnership-on-AI), [the embrace](https://www.youtube.com/watch?v=EAw_Kfg0qoo) of [more remote](https://finance.yahoo.com/news/apple-missing-ai-hype-140002045.html) [overtones](https://www.pymnts.com/artificial-intelligence-2/2024/can-apple-rely-on-its-vast-user-base-give-it-an-ai-edge/) of the [harmonic](https://www.reddit.com/r/singularity/comments/1b34dmf/do_you_think_apple_will_be_left_behind_in_the_ai/?rdt=61575) [series](https://www.wired.com/story/apple-ghosts-the-generative-ai-revolution/) - ***Q**ualities*

- [Westmalle Dubbel](https://www.youtube.com/watch?v=r3De5ji6QsY), $7$ %
- [Duvel](https://www.economist.com/business/2024/03/03/apple-is-right-not-to-rush-headlong-into-generative-ai), $8.5$ %
- [Truth](https://www.mindstream.news/p/apple-missed-ai-boat), $8.7$ %
- [Westmalle Tripel](https://medium.com/@ignacio.de.gregorio.noblejas/apple-might-have-a-real-ai-problem-920d55a2732f), $9.5$ %
- [St. Bernardus](https://www.wsj.com/tech/ai/apple-investors-grow-impatient-on-artificial-intelligence-3f934e1e) `Abt 12`, $10$ %

![](https://abikesa.github.io/belgian/craft.png)


