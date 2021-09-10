---
layout: page
permalink: /news/
title: news
description: 
nav: true
---

<div class="news">
    <div class="table-responsive">
      <table class="table table-sm table-borderless">
      {% assign news = site.news | reverse %}
      {% for item in news %}
        <tr>
          <th scope="row">{{ item.date | date: "%b %Y" }}</th>
          <td>
            {% if item.inline %}
              {{ item.content | remove: '<p>' | remove: '</p>' | emojify }}
            {% else %}
              <a class="news-title" href="{{ item.url | relative_url }}">{{ item.title }}</a>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </table>
    </div>
</div>

 