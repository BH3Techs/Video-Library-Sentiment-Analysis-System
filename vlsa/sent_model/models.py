from django.db import models
from django.utils import timezone


class Post(models.Model):
    date_posted = models.DateTimeField(default=timezone.now)
    sentiment = models.TextField()