$(document).ready(
    $(window).scroll(function() {    
    var scroll = $(window).scrollTop();

    if (scroll >= 500) {
        $(".scroll-img").css("opacity", 1);
    } else {
        $(".scroll-img").css("opacity", 0);
    }
    });
);