---
layout: page
permalink: /publications/
title: publications
description: For a complete list, please check out my [Google Scholar page](https://scholar.google.com/citations?user=BZ0EoqIAAAAJ).
years: [2021, 2020, 2019]
nav: true
---

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
 