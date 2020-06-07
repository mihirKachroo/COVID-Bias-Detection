[].forEach.call(document.querySelectorAll(".mdc-tab-bar"), function (el) {
  mdc.tabBar.MDCTabBar.attachTo(el);
});
