
  <img src="static/fulllogo.png" alt="logo" style="height: 80px; display: flex;">



<img src="https://drive.google.com/uc?export=view&id=1_-yHBRthFnKFq8l_MFRfYM-6oWvKNbco" alt="Homograph" width="45%">
We aim to tackle the issue of homograph attacks on internationalized domain names. These attacks often involve subtle modifications, such as altering individual characters or the font used in the domain name. By creating these deceptive domain names, scammers exploit unknowing users via email and various other communication methods. Here is an example bellow.
<img src="https://drive.google.com/uc?export=view&id=1YQ1FDz7DeF5LCPIXZTO4siiOBsuEDvkG" alt="Homograph" width="45%">

## Data Collection and Parsing 
List of Top 1 Million Domains: <a href="https://github.com/mozilla/cipherscan/tree/master/top1m">
    <img src="https://icones.pro/wp-content/uploads/2021/06/cliquez-sur-l-icone-violet.png" alt="Click Icon" style="width: 5%;">
</a>




List of Malicious Domains: <a href="https://cert.pl/en/posts/2020/03/malicious_domains/">
    <img src="https://icones.pro/wp-content/uploads/2021/06/cliquez-sur-l-icone-violet.png" alt="Click Icon" style="width: 5%;">
</a>


We collected data for training a decision tree and an ID3 algorithm to detect homograph attacks by sourcing valid domain names from the "List of Top 1 Million Domains" provided by Mozilla and identifying malicious domains from the "List of Malicious Domains" published by CERT Poland. This approach allowed us to compile a dataset of labeled examples, categorizing each domain as either legitimate or a homograph attack, thereby setting the groundwork for supervised learning.

For feature extraction, we analyze each domain name to identify characteristics that are indicative of legitimate or malicious intent. This includes measuring the domain name's length, identifying the presence of unusual character combinations, assessing the use of internationalized domain names (IDNs), and calculating character entropy. These features are selected because they are likely to reveal patterns that differentiate genuine domains from their homographed counterparts.

Once the features are extracted, they will be passed to both the decision tree and ID3 algorithms. These algorithms will then utilize the labeled dataset to learn the distinguishing patterns of legitimate and malicious domains. The learning process involves iteratively splitting the dataset based on the feature that maximizes information gain at each step, ultimately constructing a model that can classify new, unlabeled domain names as either legitimate or indicative of a homograph attack based on the identified patterns.
