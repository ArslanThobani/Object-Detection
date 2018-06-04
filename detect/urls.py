from django.conf import settings
from django.conf.urls.static import static
from detect import views as uploader_views
from django.urls import path
from django.conf.urls import url
from django.views.generic import TemplateView

urlpatterns = [
    path('upload', uploader_views.home, name='imageupload'),
    url('new',uploader_views.new),
    url('result', TemplateView.as_view(template_name = "result.html"))
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)