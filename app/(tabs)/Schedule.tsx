import {
  View,
  Text,
  StyleSheet,
  Alert,
  TouchableOpacity,
  FlatList,
  Image,
  Linking,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Modal
} from "react-native";
import axios from "axios";
import React, { useState, useEffect } from "react";
import { useLocalSearchParams, useRouter } from "expo-router";
import * as Notifications from "expo-notifications";
import * as Speech from "expo-speech";
import { Feather } from "@expo/vector-icons";
import DateTimePicker from "@react-native-community/datetimepicker";
import Slider from "@react-native-community/slider";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useLanguage } from "@/components/LanguageContext";
import translations from "@/components/translation";

const apollo = require("../../assets/images/apollo.png");
const medplus = require("../../assets/images/medplus.png");
const netmeds = require("../../assets/images/netmeds.png");
const onemg = require("../../assets/images/onemg.png");
const pharmeasy = require("../../assets/images/pharmeasy.png");

// Configure notification handler with proper settings
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// Register for push notifications with better error handling
export async function registerForPushNotificationsAsync(): Promise<string | null> {
  let token: string | null = null;

  try {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== "granted") {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== "granted") {
      console.log("Notification permission not granted");
      return null;
    }

    try {
      const response = await Notifications.getExpoPushTokenAsync({
        projectId: undefined, // Add your Expo project ID here if you have one
      });
      token = response.data;

      if (Platform.OS === "android") {
        await Notifications.setNotificationChannelAsync("default", {
          name: "default",
          importance: Notifications.AndroidImportance.MAX,
          vibrationPattern: [0, 250, 250, 250],
          lightColor: "#1a8e2d",
        });
      }

      return token;
    } catch (error) {
      console.error("Error getting push token:", error);
      return null;
    }
  } catch (error) {
    console.error("Error in notification registration:", error);
    return null;
  }
}

// Register notification categories for action buttons with error handling
async function registerNotificationCategories() {
  try {
    if (Platform.OS === 'ios') {
      await Notifications.setNotificationCategoryAsync('medicine', [
        {
          identifier: 'TAKE_MEDICINE',
          buttonTitle: 'Mark as Taken',
          options: {
            isDestructive: false,
            isAuthenticationRequired: false,
          },
        },
      ]);
    }
  } catch (error) {
    console.error("Failed to register notification categories:", error);
  }
}

// Schedule immediate notification for 2 minutes from now
async function scheduleImmediateReminder(medicineName: string): Promise<string | undefined> {
  try {
    // Set trigger for 2 minutes from now
    const twoMinutesFromNow = new Date();
    twoMinutesFromNow.setMinutes(twoMinutesFromNow.getMinutes() + 2);
    
    console.log(`Scheduling immediate reminder for ${medicineName} at ${twoMinutesFromNow.toLocaleTimeString()}`);
    
    // Create the trigger
    const trigger = new Date(twoMinutesFromNow);
    
    // Prepare notification content
    const content: Notifications.NotificationContentInput = {
      title: `Time to take ${medicineName}`,
      body: `Your reminder for ${medicineName}`,
      data: { medicationId: `immediate_${Date.now()}`, type: 'immediate' },
      sound: true,
    };
    
    // Add category for iOS
    if (Platform.OS === 'ios') {
      content.categoryIdentifier = "medicine";
    }
    
    // Schedule the notification
    const identifier = await Notifications.scheduleNotificationAsync({
      content,
      trigger,
    });
    
    console.log(`Immediate notification scheduled with ID: ${identifier} for ${twoMinutesFromNow.toLocaleTimeString()}`);
    return identifier;
  } catch (error) {
    console.error("Error scheduling immediate notification:", error);
    return undefined;
  }
}

// Interface for medication reminder data
interface MedicationReminder {
  id: string;
  medicineName: string;
  scheduledTime: string;
  day: number;
  taken: boolean;
  reminderEnabled?: boolean;
  times?: string[];
  notificationId?: string;
}

// Interface for medicine history data
interface MedicineHistoryItem {
  id: string;
  medicineName: string;
  medicineInfo: string;
  numberOfDays: number;
  reminderTime: string;
  createdAt: string;
  notificationIds: string[];
}

// AsyncStorage keys
const MEDICINE_HISTORY_KEY = '@medicine_history';

export default function Schedule() {
  const params = useLocalSearchParams();
  const router = useRouter();
  const { raw_response } = params;
  const { language } = useLanguage();
  const t = (key: string) => translations[language]?.[key] || key;
  
  const [days, setDays] = useState(1);
  const [reminderTime, setReminderTime] = useState(new Date());
  const [showPicker, setShowPicker] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [medicineName, setMedicineName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [notificationIds, setNotificationIds] = useState<string[]>([]);
  const [pushToken, setPushToken] = useState<string | null>(null);
  const [hasNotificationPermission, setHasNotificationPermission] = useState(false);
  const [immediatePending, setImmediatePending] = useState(false);
  
  // New state for history functionality
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [medicineHistory, setMedicineHistory] = useState<MedicineHistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  // AsyncStorage functions
  const saveMedicineToHistory = async (medicineData: MedicineHistoryItem) => {
    try {
      const existingHistory = await AsyncStorage.getItem(MEDICINE_HISTORY_KEY);
      let historyArray: MedicineHistoryItem[] = [];
      
      if (existingHistory) {
        historyArray = JSON.parse(existingHistory);
      }
      
      // Add new medicine data to the beginning of the array
      historyArray.unshift(medicineData);
      
      // Keep only the last 50 entries to prevent storage bloat
      if (historyArray.length > 50) {
        historyArray = historyArray.slice(0, 50);
      }
      
      await AsyncStorage.setItem(MEDICINE_HISTORY_KEY, JSON.stringify(historyArray));
      console.log("Medicine data saved to history successfully");
    } catch (error) {
      console.error("Error saving medicine to history:", error);
    }
  };

  const loadMedicineHistory = async () => {
    try {
      setLoadingHistory(true);
      const historyData = await AsyncStorage.getItem(MEDICINE_HISTORY_KEY);
      
      if (historyData) {
        const parsedHistory: MedicineHistoryItem[] = JSON.parse(historyData);
        setMedicineHistory(parsedHistory);
      } else {
        setMedicineHistory([]);
      }
    } catch (error) {
      console.error("Error loading medicine history:", error);
      Alert.alert(t("Error"), t("Failed to load medicine history"));
    } finally {
      setLoadingHistory(false);
    }
  };

  const clearMedicineHistory = async () => {
    try {
      Alert.alert(
        t("Clear History"),
        t("Are you sure you want to clear all medicine history? This action cannot be undone."),
        [
          {
            text: t("Cancel"),
            style: "cancel"
          },
          {
            text: t("Clear"),
            style: "destructive",
            onPress: async () => {
              await AsyncStorage.removeItem(MEDICINE_HISTORY_KEY);
              setMedicineHistory([]);
              Alert.alert(t("Success"), t("Medicine history cleared successfully"));
            }
          }
        ]
      );
    } catch (error) {
      console.error("Error clearing medicine history:", error);
      Alert.alert(t("Error"), t("Failed to clear medicine history"));
    }
  };

  const deleteSingleHistoryItem = async (itemId: string) => {
    try {
      const existingHistory = await AsyncStorage.getItem(MEDICINE_HISTORY_KEY);
      if (existingHistory) {
        let historyArray: MedicineHistoryItem[] = JSON.parse(existingHistory);
        historyArray = historyArray.filter(item => item.id !== itemId);
        await AsyncStorage.setItem(MEDICINE_HISTORY_KEY, JSON.stringify(historyArray));
        setMedicineHistory(historyArray);
      }
    } catch (error) {
      console.error("Error deleting history item:", error);
      Alert.alert(t("Error"), t("Failed to delete history item"));
    }
  };

  // Improved medicine name extraction function
  const extractMedicineNameFromResponse = (responseText: string): string => {
    if (!responseText) return "Medicine";
    
    // Look for common patterns where medicine names might appear
    const patterns = [
      /medicine name:?\s*([A-Za-z0-9\s\-]+)/i,
      /prescribed:?\s*([A-Za-z0-9\s\-]+)/i,
      /medication:?\s*([A-Za-z0-9\s\-]+)/i,
      /drug:?\s*([A-Za-z0-9\s\-]+)/i,
      /tablet:?\s*([A-Za-z0-9\s\-]+)/i,
      /capsule:?\s*([A-Za-z0-9\s\-]+)/i
    ];
    
    for (const pattern of patterns) {
      const match = responseText.match(pattern);
      if (match && match[1]) {
        return match[1].trim();
      }
    }
    
    // If no specific pattern is found, try to extract the first capitalized word
    const words = responseText.split(/\s+/);
    for (const word of words) {
      if (word.length > 3 && 
          /^[A-Z][a-z]+$/.test(word) && 
          !["This", "That", "The", "Your", "For", "When", "Take"].includes(word)) {
        return word;
      }
    }
    
    return "Medicine"; // Fallback to generic name if nothing is found
  };

  // Setup on component mount with better permission checking
  useEffect(() => {
    const setupNotifications = async () => {
      try {
        await registerNotificationCategories();
        
        // Check notification permissions first
        const { status } = await Notifications.getPermissionsAsync();
        setHasNotificationPermission(status === 'granted');
        
        if (status !== 'granted') {
          // Request permission if not granted
          const { status: newStatus } = await Notifications.requestPermissionsAsync();
          setHasNotificationPermission(newStatus === 'granted');
          
          if (newStatus !== 'granted') {
            console.log("Notification permissions not granted");
            // We don't alert here anymore since we'll still allow storing reminders without notifications
          }
        }
        
        const token = await registerForPushNotificationsAsync();
        setPushToken(token);
        
        if (!token) {
          console.log("Failed to get push notification token");
        } else {
          console.log("Push notification token:", token);
        }
      } catch (error) {
        console.error("Error in setupNotifications:", error);
      }
    };
    
    setupNotifications();
  }, [t]);

  // Extract medicine name from the response
  useEffect(() => {
    if (raw_response) {
      try {
        const extractedName = extractMedicineNameFromResponse(raw_response.toString());
        setMedicineName(extractedName || "Medicine");
      } catch (error) {
        console.error("Error extracting medicine name:", error);
        setMedicineName("Medicine"); // Fallback to generic name
      }
    }
  }, [raw_response]);

  // Function to handle text-to-speech
  const speakText = () => {
    if (speaking) {
      Speech.stop();
      setSpeaking(false);
      return;
    }
    if (raw_response) {
      setSpeaking(true);
      Speech.speak(raw_response.toString(), {
        language: language === 'hi' ? 'hi-IN' : 'en-US',
        pitch: 1.0,
        rate: 0.9,
        onDone: () => setSpeaking(false),
        onError: () => {
          setSpeaking(false);
          Alert.alert(t("Error"), t("Failed to speak text. Please try again."));
        }
      });
    } else {
      Alert.alert(t("Error"), t("No text available to speak."));
    }
  };

  // Stop speaking when component unmounts
  useEffect(() => {
    return () => {
      if (speaking) {
        Speech.stop();
      }
    };
  }, [speaking]);

  useEffect(() => {
    console.log("Params:", params);
    if (!raw_response) {
      Alert.alert(t("Error"), t("Some required data is missing."));
    }
  }, [params, t]);

  // Setup notification response listener
  useEffect(() => {
    const subscription = Notifications.addNotificationResponseReceivedListener(response => {
      try {
        const { actionIdentifier, notification } = response;
        
        if (actionIdentifier === 'TAKE_MEDICINE') {
          // User confirmed taking medicine
          const notificationId = notification.request.identifier;
          const medicationId = notification.request.content.data?.medicationId;
          
          markMedicineTaken(notificationId, medicationId);
        }
      } catch (error) {
        console.error("Error in notification response listener:", error);
      }
    });
    
    return () => {
      subscription.remove();
    };
  }, []);

  // Check all scheduled notifications - useful for debugging
  const checkScheduledNotifications = async () => {
    try {
      const scheduled = await Notifications.getAllScheduledNotificationsAsync();
      console.log("Currently scheduled notifications:", scheduled.length);
      return scheduled;
    } catch (error) {
      console.error("Error checking scheduled notifications:", error);
      return [];
    }
  };

  // Cancel medication reminders by ID with improved error handling
  async function cancelMedicationReminders(medicationId: string): Promise<void> {
    try {
      const scheduledNotifications = await Notifications.getAllScheduledNotificationsAsync();
      
      for (const notification of scheduledNotifications) {
        try {
          const data = notification.content.data as { medicationId?: string } | null;
          if (data?.medicationId === medicationId) {
            await Notifications.cancelScheduledNotificationAsync(notification.identifier);
            console.log(`Cancelled notification: ${notification.identifier}`);
          }
        } catch (cancelError) {
          console.error("Error canceling specific notification:", cancelError);
          // Continue with other notifications
        }
      }
    } catch (error) {
      console.error("Error canceling medication reminders:", error);
    }
  }

  // Function to mark medicine as taken with improved error handling
  const markMedicineTaken = async (notificationId: string, medicationId?: string) => {
    try {
      // Get the reminderId from the notification data
      let reminderId = medicationId;
      
      try {
        const scheduledNotification = await Notifications.getLastNotificationResponseAsync();
        reminderId = scheduledNotification?.notification.request.content.data?.reminderId || medicationId;
      } catch (responseError) {
        console.log("Error getting notification response:", responseError);
        // Continue with medicationId if available
      }
      
      if (!reminderId) {
        console.log("Reminder ID not found in notification data");
        // Still try to cancel the notification
        try {
          await Notifications.cancelScheduledNotificationAsync(notificationId);
        } catch (cancelError) {
          console.error("Error cancelling notification:", cancelError);
        }
        return;
      }
      
      // Check if this is an immediate notification
      const isImmediate = reminderId.toString().startsWith('immediate_');
      if (isImmediate) {
        // Just cancel the notification and show success message
        try {
          await Notifications.cancelScheduledNotificationAsync(notificationId);
          Alert.alert(t("Success"), t("Medicine marked as taken!"));
        } catch (cancelError) {
          console.error("Error cancelling immediate notification:", cancelError);
        }
        return;
      }
      
      // Make API call to mark medication as taken
      try {
        await axios.post('/api/medicine/taken', {
          reminderId: reminderId,
          notificationId: notificationId
        });
        
        // Cancel the notification
        await Notifications.cancelScheduledNotificationAsync(notificationId);
        
        // Show confirmation to user
        Alert.alert(t("Success"), t("Medicine marked as taken!"));
      } catch (apiError) {
        console.error("API call failed:", apiError);
        
        // Even if API fails, still cancel the notification
        try {
          await Notifications.cancelScheduledNotificationAsync(notificationId);
        } catch (cancelError) {
          console.error("Error cancelling notification after API error:", cancelError);
        }
        
        Alert.alert(t("Partial Success"), t("Medicine marked as taken, but server update failed."));
      }
    } catch (error) {
      console.error("Failed to mark medicine as taken:", error);
      Alert.alert(t("Error"), t("Failed to mark medicine as taken. Please try again."));
    }
  };

  // Improved schedule medication reminder function - now just focuses on scheduling notifications
  async function scheduleMedicationReminder(medication: MedicationReminder): Promise<string | undefined> {
    if (!hasNotificationPermission) {
      console.log("No notification permission");
      return undefined;
    }
    
    try {
      const scheduledTime = new Date(medication.scheduledTime);
      const hours = scheduledTime.getHours();
      const minutes = scheduledTime.getMinutes();
      
      // Format time for logging
      const timeStr = `${hours}:${minutes < 10 ? '0' + minutes : minutes}`;
      console.log(`Scheduling reminder for ${medication.medicineName} at ${timeStr}`);
      
      // Create trigger based on hour and minute for daily notifications
      const trigger: Notifications.NotificationTriggerInput = {
        hour: hours,
        minute: minutes,
        repeats: false, // No repeat for one-time reminders
      };
      
      // Set the date for the trigger if scheduling for future days
      const currentDate = new Date();
      if (scheduledTime.getDate() !== currentDate.getDate() || 
          scheduledTime.getMonth() !== currentDate.getMonth() || 
          scheduledTime.getFullYear() !== currentDate.getFullYear()) {
        
        // Add date components to the trigger
        trigger.day = scheduledTime.getDate();
        trigger.month = scheduledTime.getMonth() + 1; // Months are 0-indexed in JS
        trigger.year = scheduledTime.getFullYear();
      }
      
      // Prepare notification content
      const content: Notifications.NotificationContentInput = {
        title: t("Medication Reminder"),
        body: t(`Time to take ${medication.medicineName}`),
        data: { medicationId: medication.id, reminderId: medication.id },
        sound: true,
      };
      
      // Add category for iOS
      if (Platform.OS === 'ios') {
        content.categoryIdentifier = "medicine";
      }
      
      // Schedule the notification with timeout protection
      const schedulePromise = Notifications.scheduleNotificationAsync({
        content,
        trigger,
      });
      
      // Create a timeout promise
      const timeoutPromise = new Promise<string>((_, reject) => {
        setTimeout(() => reject(new Error("Notification scheduling timed out")), 5000);
      });
      
      // Race the promises
      const identifier = await Promise.race([schedulePromise, timeoutPromise]);
      
      console.log(`Notification scheduled with ID: ${identifier}`);
      return identifier;
    } catch (error) {
      console.error("Error scheduling medication reminder:", error);
      return undefined;
    }
  }

  // New function to save medication reminder to database regardless of notification status
  const saveMedicationReminder = async (reminderObj: MedicationReminder): Promise<string | null> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 7000);
      
      // Save to database
      const apiUrl = '/api/medicine/schedule';
      const response = await axios.post(apiUrl, reminderObj, {
        signal: controller.signal,
        timeout: 7000
      });
      
      clearTimeout(timeoutId);
      
      if (response && response.data && response.data.id) {
        console.log(`Reminder saved to database with ID: ${response.data.id}`);
        return response.data.id;
      } else {
        console.error("Invalid response from server when saving reminder");
        return null;
      }
    } catch (error) {
      console.error("Failed to save medication reminder to database:", error);
      return null;
    }
  };

  // Update notification ID for a reminder in the database
  const updateReminderWithNotificationId = async (reminderId: string, notificationId: string): Promise<boolean> => {
    try {
      await axios.put(`/api/medicine/schedule/${reminderId}`, {
        notificationId
      }, { timeout: 5000 });
      
      console.log(`Updated reminder ${reminderId} with notification ID ${notificationId}`);
      return true;
    } catch (error) {
      console.error(`Failed to update reminder ${reminderId} with notification ID:`, error);
      return false;
    }
  };

  // Schedule reminders with improved error handling - separated database save from notification scheduling
  const scheduleReminders = async () => {
    if (!medicineName) {
      Alert.alert(t("Error"), t("Could not determine medicine name. Please try again."));
      return;
    }

    setIsLoading(true);
    setImmediatePending(true);
    const currentTime = new Date();
    const savedReminderIds = [];
    const newNotificationIds = [];
    const errorMessages = [];
    let successfulSaves = 0;
    let successfulNotifications = 0;
    let immediateNotificationId: string | undefined;

    try {
      // First, schedule the immediate notification for 2 minutes from now
      if (hasNotificationPermission) {
        immediateNotificationId = await scheduleImmediateReminder(medicineName);
        if (immediateNotificationId) {
          newNotificationIds.push(immediateNotificationId);
          console.log("Immediate reminder scheduled successfully");
        } else {
          console.error("Failed to schedule immediate notification");
        }
      }

      // Cancel any existing notifications for regular reminders
      if (notificationIds.length > 0) {
        for (const id of notificationIds) {
          try {
            await Notifications.cancelScheduledNotificationAsync(id);
          } catch (cancelError) {
            console.log("Error cancelling notification:", cancelError);
            // Continue even if cancellation fails
          }
        }
      }

      // Check current scheduled notifications
      await checkScheduledNotifications();

      // Create and save all reminders first
      for (let i = 0; i < days; i++) {
        try {
          // Create a new date for this reminder
          const reminderDate = new Date();
          reminderDate.setDate(currentTime.getDate() + i);
          reminderDate.setHours(reminderTime.getHours());
          reminderDate.setMinutes(reminderTime.getMinutes());
          reminderDate.setSeconds(0);
          reminderDate.setMilliseconds(0);
          
          // Ensure time is in the future for today's reminder
          if (i === 0 && reminderDate <= currentTime) {
            reminderDate.setDate(reminderDate.getDate() + 1);
          }
          
          console.log(`Creating reminder for: ${reminderDate.toString()}`);
          
          // Create reminder object
          const reminderObj: MedicationReminder = {
            id: `med_${Date.now()}_${i}`, // Generate a unique ID
            medicineName,
            scheduledTime: reminderDate.toISOString(),
            day: i + 1,
            taken: false,
            reminderEnabled: true,
            times: [`${reminderTime.getHours()}:${reminderTime.getMinutes() < 10 ? '0' + reminderTime.getMinutes() : reminderTime.getMinutes()}`]
          };

          // Save the reminder to the database first - this is now decoupled from notification scheduling
          const savedId = await saveMedicationReminder(reminderObj);
          
          if (savedId) {
            // Update with the database ID
            reminderObj.id = savedId;
            savedReminderIds.push(savedId);
            successfulSaves++;
            
            // Now try to schedule notification if permission exists
            if (hasNotificationPermission) {
              const notificationId = await scheduleMedicationReminder(reminderObj);
              
              if (notificationId) {
                // Successfully scheduled notification, update in database
                await updateReminderWithNotificationId(savedId, notificationId);
                newNotificationIds.push(notificationId);
                successfulNotifications++;
              } else {
                console.log(`Failed to schedule notification for day ${i+1}, but reminder is saved`);
              }
            }
          } else {
            errorMessages.push(`Failed to save reminder for day ${i + 1}`);
          }
        } catch (dayError) {
          console.error(`Error processing day ${i + 1}:`, dayError);
          errorMessages.push(`Error on day ${i + 1}: ${dayError.message}`);
          // Continue with next day
        }
      }

      // Update notification IDs state - include both immediate and regular notifications
      setNotificationIds(newNotificationIds);
      
      // Save medicine data to AsyncStorage history
      const historyItem: MedicineHistoryItem = {
        id: `history_${Date.now()}`,
        medicineName,
        medicineInfo: raw_response?.toString() || '',
        numberOfDays: days,
        reminderTime: reminderTime.toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit'
        }),
        createdAt: new Date().toISOString(),
        notificationIds: newNotificationIds
      };
      
      await saveMedicineToHistory(historyItem);
      
      // Show appropriate message based on results
      let successMessage;
      if (successfulSaves === 0 && !immediateNotificationId) {
        // No reminders were saved and no immediate notification
        throw new Error("Failed to save any reminders to the database");
      } else {
        // Determine message based on what succeeded
        if (immediateNotificationId) {
          successMessage = t("Reminder Scheduled Successfully");
          if (successfulSaves > 0) {
            successMessage += `\n${t("{{count}} reminders have been scheduled for {{medicine}}").replace("{{count}}", days.toString()).replace("{{medicine}}", medicineName)}`;
          }
        } else if (successfulSaves > 0) {
          successMessage = t("{{count}} reminders have been scheduled for {{medicine}}").replace("{{count}}", days.toString()).replace("{{medicine}}", medicineName);
        }
        
        Alert.alert(t("Reminders Set"), successMessage, [{ text: "OK" }]);
      }
    } catch (error) {
      console.error("Failed to schedule reminders:", error);
      
      // Create detailed error message
      let errorMsg = t("Error scheduling reminders.");
      if (errorMessages.length > 0) {
        errorMsg += "\n\n" + errorMessages.slice(0, 3).join("\n");
        if (errorMessages.length > 3) {
          errorMsg += `\n... and ${errorMessages.length - 3} more`;
        }
      }
      
      if (successfulSaves > 0 || immediateNotificationId) {
        // Some reminders were saved despite errors
        errorMsg += `\n\n${t("However, some reminders were saved successfully.")}`;
        if (immediateNotificationId) {
          errorMsg += `\n${t("Immediate reminder set for 2 minutes from now.")}`;
        }
      }
      
      Alert.alert(t("Error"), errorMsg);
    } finally {
      setIsLoading(false);
      setImmediatePending(false);
    }
  };

  // Function to handle opening history modal
  const openHistoryModal = async () => {
    setShowHistoryModal(true);
    await loadMedicineHistory();
  };

  // Function to format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Render history item
  const renderHistoryItem = ({ item }: { item: MedicineHistoryItem }) => (
    <View style={styles.historyItem}>
      <View style={styles.historyItemHeader}>
        <Text style={styles.historyMedicineName}>{item.medicineName}</Text>
        <TouchableOpacity
          style={styles.deleteHistoryButton}
          onPress={() => {
            Alert.alert(
              t("Delete Item"),
              t("Are you sure you want to delete this history item?"),
              [
                { text: t("Cancel"), style: "cancel" },
                {
                  text: t("Delete"),
                  style: "destructive",
                  onPress: () => deleteSingleHistoryItem(item.id)
                }
              ]
            );
          }}
        >
          <Feather name="trash-2" size={16} color="#ff4444" />
        </TouchableOpacity>
      </View>
      <Text style={styles.historyInfo} numberOfLines={50}>
        {item.medicineInfo.length > 100 
          ? item.medicineInfo.substring(0, 100) + '...' 
          : item.medicineInfo}
      </Text>
      <View style={styles.historyDetails}>
        <Text style={styles.historyDetailText}>
          {t("Days")}: {item.numberOfDays}
        </Text>
        <Text style={styles.historyDetailText}>
          {t("Time")}: {item.reminderTime}
        </Text>
      </View>
      <Text style={styles.historyDate}>
        {t("Created")}: {formatDate(item.createdAt)}
      </Text>
    </View>
  );

  const handleTimeChange = (event: any, selectedTime?: Date) => {
    const currentTime = selectedTime || reminderTime;
    setShowPicker(Platform.OS === 'ios');
    setReminderTime(currentTime);
  };

  const openPharmacyApp = (url: string) => {
    Linking.openURL(url).catch((err) => {
      console.error("Failed to open URL:", err);
      Alert.alert(t("Error"), t("Could not open the pharmacy app"));
    });
  };

  if (!raw_response) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>{t("No medicine information available")}</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        {/* Header with History Button */}
        <View style={styles.header}>
          <Text style={styles.title}>{t("Medicine Schedule")}</Text>
          <TouchableOpacity
            style={styles.historyButton}
            onPress={openHistoryModal}
          >
            <Feather name="clock" size={20} color="#4A90E2" />
            <Text style={styles.historyButtonText}>{t("History")}</Text>
          </TouchableOpacity>
        </View>

        {/* Medicine Information Card */}
        <View style={styles.medicineCard}>
          <View style={styles.medicineHeader}>
            <Text style={styles.medicineName}>{medicineName}</Text>
            <TouchableOpacity
              style={[styles.speakButton, speaking && styles.speakButtonActive]}
              onPress={speakText}
            >
              <Feather 
                name={speaking ? "volume-x" : "volume-2"} 
                size={20} 
                color={speaking ? "#ff4444" : "#4A90E2"} 
              />
            </TouchableOpacity>
          </View>
          <ScrollView style={styles.responseContainer} nestedScrollEnabled={true}>
            <Text style={styles.responseText}>{raw_response}</Text>
          </ScrollView>
        </View>

        {/* Schedule Settings */}
        <View style={styles.scheduleCard}>
          <Text style={styles.sectionTitle}>{t("Schedule Settings")}</Text>
          
          {/* Days Slider */}
          <View style={styles.sliderContainer}>
            <Text style={styles.sliderLabel}>
              {t("Number of days")}: {days}
            </Text>
            <Slider
              style={styles.slider}
              minimumValue={1}
              maximumValue={30}
              value={days}
              onValueChange={(value) => setDays(Math.round(value))}
              minimumTrackTintColor="#4A90E2"
              maximumTrackTintColor="#d3d3d3"
              thumbStyle={styles.sliderThumb}
            />
          </View>

          {/* Time Picker */}
          <View style={styles.timeContainer}>
            <Text style={styles.timeLabel}>{t("Reminder Time")}</Text>
            <TouchableOpacity
              style={styles.timeButton}
              onPress={() => setShowPicker(true)}
            >
              <Text style={styles.timeButtonText}>
                {reminderTime.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </Text>
              <Feather name="clock" size={20} color="#4A90E2" />
            </TouchableOpacity>
          </View>

          {showPicker && (
            <DateTimePicker
              testID="dateTimePicker"
              value={reminderTime}
              mode="time"
              is24Hour={false}
              display="default"
              onChange={handleTimeChange}
            />
          )}

          {/* Notification Status */}
          {!hasNotificationPermission && (
            <View style={styles.warningContainer}>
              <Feather name="alert-triangle" size={16} color="#ff8800" />
              <Text style={styles.warningText}>
                {t("Notification permission not granted. Reminders will be saved but no notifications will be shown.")}
              </Text>
            </View>
          )}

          {/* Immediate Reminder Status */}
          {immediatePending && (
            <View style={styles.infoContainer}>
              <Feather name="info" size={16} color="#4A90E2" />
              <Text style={styles.infoText}>
                {t("An immediate reminder will be set for 2 minutes from now.")}
              </Text>
            </View>
          )}

          {/* Schedule Button */}
          <TouchableOpacity
            style={[styles.scheduleButton, isLoading && styles.scheduleButtonDisabled]}
            onPress={scheduleReminders}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#ffffff" size="small" />
            ) : (
              <Text style={styles.scheduleButtonText}>
                {t("Set Reminders")}
              </Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Pharmacy Apps Section */}
        <View style={styles.pharmacyCard}>
          <Text style={styles.sectionTitle}>{t("Order Medicine")}</Text>
          <Text style={styles.pharmacySubtitle}>
            {t("Choose from these trusted pharmacy apps")}
          </Text>
          
          <View style={styles.pharmacyGrid}>
            <TouchableOpacity
              style={styles.pharmacyItem}
              onPress={() => openPharmacyApp("https://www.1mg.com/")}
            >
              <Image source={onemg} style={styles.pharmacyLogo} />
              <Text style={styles.pharmacyName}>1mg</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.pharmacyItem}
              onPress={() => openPharmacyApp("https://pharmeasy.in/")}
            >
              <Image source={pharmeasy} style={styles.pharmacyLogo} />
              <Text style={styles.pharmacyName}>PharmEasy</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.pharmacyItem}
              onPress={() => openPharmacyApp("https://www.netmeds.com/")}
            >
              <Image source={netmeds} style={styles.pharmacyLogo} />
              <Text style={styles.pharmacyName}>Netmeds</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.pharmacyItem}
              onPress={() => openPharmacyApp("https://www.medplusmart.com/")}
            >
              <Image source={medplus} style={styles.pharmacyLogo} />
              <Text style={styles.pharmacyName}>MedPlus</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.pharmacyItem}
              onPress={() => openPharmacyApp("https://www.apollopharmacy.in/")}
            >
              <Image source={apollo} style={styles.pharmacyLogo} />
              <Text style={styles.pharmacyName}>Apollo</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>

      {/* History Modal */}
      <Modal
        visible={showHistoryModal}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setShowHistoryModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>{t("Medicine History")}</Text>
            <View style={styles.modalHeaderButtons}>
              {medicineHistory.length > 0 && (
                <TouchableOpacity
                  style={styles.clearHistoryButton}
                  onPress={clearMedicineHistory}
                >
                  <Feather name="trash" size={18} color="#ff4444" />
                  <Text style={styles.clearHistoryText}>{t("Clear All")}</Text>
                </TouchableOpacity>
              )}
              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setShowHistoryModal(false)}
              >
                <Feather name="x" size={24} color="#333" />
              </TouchableOpacity>
            </View>
          </View>

          {loadingHistory ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#4A90E2" />
              <Text style={styles.loadingText}>{t("Loading history...")}</Text>
            </View>
          ) : medicineHistory.length === 0 ? (
            <View style={styles.emptyHistoryContainer}>
              <Feather name="clock" size={48} color="#ccc" />
              <Text style={styles.emptyHistoryText}>
                {t("No medicine history found")}
              </Text>
              <Text style={styles.emptyHistorySubtext}>
                {t("Schedule some medicines to see them here")}
              </Text>
            </View>
          ) : (
            <FlatList
              data={medicineHistory}
              renderItem={renderHistoryItem}
              keyExtractor={(item) => item.id}
              style={styles.historyList}
              showsVerticalScrollIndicator={false}
            />
          )}
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  scrollContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  historyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  historyButtonText: {
    marginLeft: 6,
    color: '#4A90E2',
    fontWeight: '600',
  },
  medicineCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  medicineHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  medicineName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    flex: 1,
  },
  speakButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: '#f8f9fa',
  },
  speakButtonActive: {
    backgroundColor: '#ffe6e6',
  },
  responseContainer: {
    maxHeight: 150,
  },
  responseText: {
    fontSize: 14,
    lineHeight: 20,
    color: '#555',
  },
  scheduleCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 16,
  },
  sliderContainer: {
    marginBottom: 20,
  },
  sliderLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#34495e',
    marginBottom: 10,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  sliderThumb: {
    backgroundColor: '#4A90E2',
    width: 20,
    height: 20,
  },
  timeContainer: {
    marginBottom: 20,
  },
  timeLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#34495e',
    marginBottom: 10,
  },
  timeButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  timeButtonText: {
    fontSize: 16,
    color: '#2c3e50',
    fontWeight: '500',
  },
  warningContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3cd',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  warningText: {
    marginLeft: 8,
    fontSize: 12,
    color: '#856404',
    flex: 1,
  },
  infoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#d1ecf1',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  infoText: {
    marginLeft: 8,
    fontSize: 12,
    color: '#0c5460',
    flex: 1,
  },
  scheduleButton: {
    backgroundColor: '#4A90E2',
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  scheduleButtonDisabled: {
    backgroundColor: '#bdc3c7',
  },
  scheduleButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  pharmacyCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  pharmacySubtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 16,
  },
  pharmacyGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  pharmacyItem: {
    width: '30%',
    alignItems: 'center',
    marginBottom: 16,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#f8f9fa',
  },
  pharmacyLogo: {
    width: 40,
    height: 40,
    marginBottom: 8,
    resizeMode: 'contain',
  },
  pharmacyName: {
    fontSize: 12,
    fontWeight: '500',
    color: '#2c3e50',
    textAlign: 'center',
  },
  errorText: {
    fontSize: 16,
    color: '#e74c3c',
    textAlign: 'center',
    marginTop: 50,
  },
  // Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    paddingTop: Platform.OS === 'ios' ? 50 : 20, // Account for status bar
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  modalHeaderButtons: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  clearHistoryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    backgroundColor: '#ffe6e6',
  },
  clearHistoryText: {
    marginLeft: 4,
    color: '#ff4444',
    fontSize: 12,
    fontWeight: '600',
  },
  closeButton: {
    padding: 4,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 100, // Account for potential bottom navigation
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#7f8c8d',
  },
  emptyHistoryContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
    paddingBottom: 100, // Account for potential bottom navigation
  },
  emptyHistoryText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#7f8c8d',
    marginTop: 16,
    textAlign: 'center',
  },
  emptyHistorySubtext: {
    fontSize: 14,
    color: '#bdc3c7',
    marginTop: 8,
    textAlign: 'center',
  },
  historyList: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: Platform.OS === 'ios' ? 34 : 16, // Account for home indicator on iOS
  },
  historyItem: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  historyItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  historyMedicineName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    flex: 1,
  },
  deleteHistoryButton: {
    padding: 8,
    borderRadius: 16,
    backgroundColor: '#ffe6e6',
  },
  historyInfo: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 8,
    lineHeight: 16,
  },
  historyDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  historyDetailText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4A90E2',
  },
  historyDate: {
    fontSize: 12,
    color: '#95a5a6',
    fontStyle: 'italic',
  },
});