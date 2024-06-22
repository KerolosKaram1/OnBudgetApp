export const getNotifications = () => {
  return [
    {
      notificationID: 1,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "Your order has been placed.",
      notificationDate: "2022-01-01 12:00:00",
      isRead: true,
    },
    {
      notificationID: 2,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "Your order has been shipped.",
      notificationDate: "2022-01-01 12:00:00",
      isRead: true,
    },
    {
      notificationID: 3,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "Your order will arrive today.",
      notificationDate: "2022-01-01 12:00:00",
      isRead: false,
    },
    {
      notificationID: 4,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "Your order is out to be delivered.",
      notificationDate: "2022-01-01 12:00:00",
      isRead: false,
    },
    {
      notificationID: 5,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "We hope that you loved our product. Please, rate it.",
      notificationDate: "2022-01-01 12:00:00",
      isRead: false,
    },
    {
      notificationID: 6,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "Don't forget to check your wishlist",
      notificationDate: "2022-01-01 12:00:00",
      isRead: false,
    },
    {
      notificationID: 7,
      userID: 12,
      notificationType: "newMessage",
      notificationText:
        "You have some items in your cart, do you wanna order them?",
      notificationDate: "2022-01-01 12:00:00",
      isRead: false,
    },
    {
      notificationID: 8,
      userID: 12,
      notificationType: "newMessage",
      notificationText: "Your order has been canceled.",
      notificationDate: "2022-01-01 12:00:00",
      isRead: false,
    },
  ];
};

export const getUnreadNotifications = () => {
  return getNotifications().filter((notification) => !notification.isRead);
};
