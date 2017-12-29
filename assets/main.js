$(window).scroll(function() {    
    var scroll = $(window).scrollTop();
    var opacity = Math.min(scroll/250, 1);
    $(".scroll-img").css("opacity", opacity);
});