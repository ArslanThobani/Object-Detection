from django.db import models
from django.forms import ModelForm
import hashlib
import datetime
import os
from functools import partial

from django.core.files.storage import FileSystemStorage
class OverwriteStorage(FileSystemStorage):
    def _save(self, name, content):
        if self.exists(name):
            self.delete(name)
        return super(OverwriteStorage, self)._save(name, content)

def _update_filename(instance, filename, path):
    path = path

    filename = "arslan.jpg"

    return os.path.join(path, filename)

def upload_to(path):
    return partial(_update_filename, path=path)

class Upload(models.Model):
    pic = models.FileField(upload_to=upload_to("images/"), storage=OverwriteStorage())
    upload_date=models.DateTimeField(auto_now_add =True)



# FileUpload form class.
class UploadForm(ModelForm):
    class Meta:
        model = Upload
        fields = ('pic',)
