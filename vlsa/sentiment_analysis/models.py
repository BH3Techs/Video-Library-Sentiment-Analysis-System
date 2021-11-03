from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


class Post(models.Model):
    date_posted = models.DateTimeField(default=timezone.now)
    sentiment = models.TextField()

    def __str__(self):
        return self.sentiment

    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'pk': self.pk})
