import {
  View,
  Text,
  StyleSheet,
  Alert,
  Button,
  FlatList,
  TouchableOpacity,
  Image,
  Linking,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator
} from "react-native";
import axios from "axios";
import React, { useState, useEffect } from "react";
import { useLocalSearchParams, useRouter } from "expo-router";
import * as Notifications from "expo-notifications";
import * as Speech from "expo-speech";
import { Feather } from "@expo/vector-icons";
import DateTimePicker from "@react-native-community/datetimepicker";
import Slider from "@react-native-community/slider";
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
    const currentTime = new Date();
    const savedReminderIds = [];
    const newNotificationIds = [];
    const errorMessages = [];
    let successfulSaves = 0;
    let successfulNotifications = 0;

    try {
      // Cancel any existing notifications first
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
                console.log(`scheduled notification for day ${i+1},reminder is saved`);
              }
            }
          } else {
            errorMessages.push(`saved reminder for day ${i + 1}`);
          }
        } catch (dayError) {
          console.error(`Error processing day ${i + 1}:`, dayError);
          errorMessages.push(`Error on day ${i + 1}: ${dayError.message}`);
          // Continue with next day
        }
      }

      // Update notification IDs state
      setNotificationIds(newNotificationIds);
      
      // Show appropriate message based on results
      if (successfulSaves === 0) {
        // No reminders were saved
        throw new Error("Failed to save any reminders to the database");
      } 
      
      if (successfulSaves === days) {
        if (hasNotificationPermission && successfulNotifications === days) {
          // Perfect scenario: all saved and all notifications scheduled
          Alert.alert(
            t("Reminders Set"), 
            t("{{count}} reminders have been scheduled for {{medicine}}").replace("{{count}}", days.toString()).replace("{{medicine}}", medicineName),
            [{ text: "OK" }]
          );
        } else if (hasNotificationPermission && successfulNotifications > 0) {
          // All saved but only some notifications scheduled
          Alert.alert(
            t("Reminders Saved"), 
            t("All {{count}} reminders saved. {{notificationCount}} notifications scheduled.").replace("{{count}}", days.toString()).replace("{{notificationCount}}", successfulNotifications.toString()),
            [{ text: "OK" }]
          );
        } else if (hasNotificationPermission) {
          // All saved but no notifications scheduled
          Alert.alert(
            t("Reminders Saved"), 
            t("All {{count}} reminders saved but notifications could not be scheduled. Please check your notification settings.").replace("{{count}}", days.toString()),
            [{ text: "OK" }]
          );
        } else {
          // All saved, no notification permission
          Alert.alert(
            t("Reminders Saved"), 
            t("All {{count}} reminders saved. Enable notifications in settings to receive alerts.").replace("{{count}}", days.toString()),
            [{ text: "OK" }]
          );
        }
      } else {
        // Some reminders saved
        Alert.alert(
          t("Partial Success"), 
          t("{{savedCount}} out of {{total}} reminders saved successfully.").replace("{{savedCount}}", successfulSaves.toString()).replace("{{total}}", days.toString()),
          [{ text: "OK" }]
        );
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
      
      if (successfulSaves > 0) {
        // Some reminders were saved despite errors
        errorMsg += `\n\n${t("However, {{count}} reminders were saved successfully.").replace("{{count}}", successfulSaves.toString())}`;
      }
      
      Alert.alert(t("Error"), errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  // Define pharmacy options
  const pharmacyOptions = [
    { id: "1", name: t("Apollo Pharmacy"), logo: apollo, url: "https://www.apollopharmacy.in/" },
    { id: "2", name: t("Med Plus"), logo: medplus, url: "https://www.medplusmart.com/" },
    { id: "3", name: t("Netmeds"), logo: netmeds, url: "https://www.netmeds.com/" },
    { id: "4", name: t("1mg"), logo: onemg, url: "https://www.1mg.com/" },
    { id: "5", name: t("PharmEasy"), logo: pharmeasy, url: "https://www.pharmeasy.in/" },
  ];

  const openWebsite = (url: string) => {
    Linking.openURL(url);
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={{ flex: 1 }}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.container}>
          <Text style={styles.title}>{t("Medicine Information")}</Text>
          {raw_response ? (
            <View style={styles.infoContainer}>
              <View style={styles.infoHeader}>
                <Text style={styles.info}>{raw_response}</Text>
                <TouchableOpacity
                  style={[
                    styles.speakerButton,
                    speaking && styles.speakerButtonActive,
                  ]}
                  onPress={speakText}
                >
                  <Feather
                    name={speaking ? "volume-2" : "volume"}
                    size={24}
                    color="#1E90FF"
                  />
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <Text>{t("No valid data available.")}</Text>
          )}
          
          {medicineName && (
            <View style={styles.medicineNameContainer}>
              <Text style={styles.medicineNameLabel}>{t("Medicine Name:")}</Text>
              <Text style={styles.medicineName}>{medicineName}</Text>
            </View>
          )}
          
          <View style={styles.shoppingContainer}>
            <Text style={styles.shoppingTitle}>
              {t("Buy Your Medicine Online")}
            </Text>
            <FlatList
              horizontal
              data={pharmacyOptions}
              keyExtractor={(item) => item.id}
              showsHorizontalScrollIndicator={false}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={styles.shoppingItem}
                  onPress={() => openWebsite(item.url)}
                >
                  <Image source={item.logo} style={styles.shoppingImage} />
                  <Text style={styles.shoppingItemText}>{item.name}</Text>
                </TouchableOpacity>
              )}
            />
          </View>

          <View style={styles.section}>
            <Text style={styles.label}>{t("Select Dosage Days:")}</Text>
            <Text style={styles.dayCount}>
              {days} {days === 1 ? t("Day") : t("Days")}
            </Text>
            <Slider
              style={styles.slider}
              value={days}
              minimumValue={1}
              maximumValue={30}
              step={1}
              onValueChange={(value) => setDays(Math.round(value))}
              minimumTrackTintColor="#1E90FF"
              maximumTrackTintColor="#d3d3d3"
              thumbTintColor="#1E90FF"
            />
          </View>

          <View style={styles.section}>
            <Text style={styles.label}>{t("Set Reminder Time:")}</Text>
            <TouchableOpacity
              style={styles.timePickerButton}
              onPress={() => setShowPicker(true)}
            >
              <Text style={styles.timePickerText}>
                {reminderTime.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </Text>
            </TouchableOpacity>

            {showPicker && (
              <DateTimePicker
                value={reminderTime}
                mode="time"
                display="default"
                onChange={(event, selectedDate) => {
                  setShowPicker(Platform.OS === 'ios' ? true : false);
                  if (selectedDate) {
                    setReminderTime(selectedDate);
                  }
                }}
              />
            )}
          </View>

          <View style={styles.buttonContainer}>
            <Button
              title={isLoading ? t("Scheduling...") : t("Schedule Reminders")}
              onPress={scheduleReminders}
              disabled={isLoading}
              color="#1E90FF"
            />
            {isLoading && (
              <ActivityIndicator
                style={styles.loadingIndicator}
                size="small"
                color="#1E90FF"
              />
            )}
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}


const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
    paddingBottom: 50,
  },
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: "#f5f5f5",
    alignItems: "center",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 10,
  },
  infoContainer: {
    padding: 15,
    backgroundColor: "#ffffff",
    borderRadius: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    marginTop: 10,
    width: "100%",
  },
  infoHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  info: {
    fontSize: 16,
    flex: 1,
  },
  medicineNameContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 15,
    backgroundColor: "#e6f2ff",
    padding: 10,
    borderRadius: 8,
    width: "100%",
  },
  medicineNameLabel: {
    fontSize: 16,
    fontWeight: "bold",
    marginRight: 5,
  },
  medicineName: {
    fontSize: 16,
    color: "#1E90FF",
    fontWeight: "500",
  },
  speakerButton: {
    marginLeft: 10,
    padding: 8,
    borderRadius: 20,
    backgroundColor: "#f0f0f0",
  },
  speakerButtonActive: {
    backgroundColor: "#e6f2ff",
  },
  shoppingContainer: {
    width: "100%",
    marginTop: 20,
  },
  shoppingTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
  },
  shoppingItem: {
    backgroundColor: "#fff",
    padding: 10,
    marginRight: 15,
    borderRadius: 8,
    alignItems: "center",
  },
  shoppingImage: {
    width: 80,
    height: 40,
    resizeMode: "contain",
  },
  shoppingItemText: {
    fontSize: 14,
    color: "#25292e",
    marginTop: 5,
  },
  section: {
    marginTop: 20,
    width: "100%",
    alignItems: "center",
  },
  label: {
    fontSize: 16,
    marginBottom: 5,
  },
  dayCount: {
    fontSize: 18,
    fontWeight: "bold",
  },
  slider: {
    width: "80%",
    height: 40,
  },
  buttonContainer: {
    marginTop: 30,
    width: "100%",
    alignItems: "center",
  },
  scheduleButton: {
    backgroundColor: "#1E90FF",
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 25,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  disabledButton: {
    backgroundColor: "#7cb7f1",
  },
  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
    marginLeft: 5,
  },
  timePickerText: {
    fontSize: 16,
    color: '#1E90FF',
  },
  loadingIndicator: {
    marginLeft: 10,
  }
});