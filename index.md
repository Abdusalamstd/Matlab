## بۇ بەتكە كەلگىنىڭىزنى قارشى ئالىمەن

تەپىسىلاتىنى مىنىڭ باش بىتىمدىن كۆرۈڭ

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### ئاددى بىر پايسون پىروگىراممىسى

كونا يېزىقنى لاتىن يېزىقىغا ئايلاندۇرۇش كودى.توغرىلىق نىسپىتى 99% كەم ئەمەس

```python
#encoding utf-8
old = [u'ا', u'ە', u'ب', u'پ', u'ت', u'ج', u'چ', u'خ', u'د', u'ر', u'ز', u'ژ', u'س', u'ش', u'ف', u'ڭ', u'ل',\
        u'م', u'ھ', u'و', u'ۇ', u'ۆ', u'ۈ', u'ۋ', u'ې', u'ى', u'ي', u'ق', u'ك', u'گ', u'ن', u'غ',u'؟']
new = [u'a', u'e',  u'b', u'p', u't', u'j', u'ch', u'x', u'd', u'r', u'z', u'j', u's', u'sh', u'f', u'ng', u'l',\
         u'm', u'h', u'o', u'u', u'ö', u'ü', u'w', u'é', u'i', u'y', u'q', u'k', u'g', u'n', u'gh',u'?']
a = open('a.txt',mode='r')
a = a.read()
b = list(a.lower())
tr=""
for i in range(0,len(b)):
    if b[i] == 'ئ':
        continue
    x = 0
    for j in range(0,len(old)):
        if b[i] == old[j]:
            tr += new[j]
            x=1
            break
    if x == 0:
        tr += b[i]
print(tr)

```

تېكسىت ھۆججىتىدە خەتلەر بولىدۇ
### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Abdusalamstd/Python/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### مەن بىلەن ئالاقىلىشىڭ

تېخىمۇ كۆپ نەرسىلەرنە سورىماقچى بىلمەكچى بولسىڭىز خەت قالدۇرۇڭ
