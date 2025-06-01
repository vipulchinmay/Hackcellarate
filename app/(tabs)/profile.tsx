import { useState, useRef, useEffect } from "react";
import {
  Text,
  View,
  TextInput,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Animated,
  KeyboardAvoidingView,
  Platform,
  Alert,
  Image,
  Easing,
  Switch,
  ActivityIndicator,
  Modal,
  FlatList,
} from "react-native";
import axios from "axios";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import { useLanguage } from "@/components/LanguageContext";
import translations from "@/components/translation";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import 'react-native-gesture-handler';
// Rest of your imports
// Base API URL for all requests
const API_URL = "http://172.20.10.5:6000";

// Interface for recent contacts
interface RecentContact {
  name: string;
  phoneNumber: string;
  relation: string;
  timestamp: number;
}

// Interface for user data
interface UserData {
  name: string;
  age: string;
  gender: string;
  blood_group: string;
  medical_conditions: string;
  date_of_birth: string;
  photo: string | null;
  _id?: string;
}

export default function ProfileScreen() {
  const { language } = useLanguage();
  const t = (key: string) => translations[language]?.[key] || key;
  const navigation = useNavigation();

  // User data state
  const [userData, setUserData] = useState<UserData>({
    name: "",
    age: "",
    gender: "",
    blood_group: "",
    medical_conditions: "",
    date_of_birth: "",
    photo: null,
  });

  // UI state variables
  const [isEditing, setIsEditing] = useState(true); // Set to true for initial registration
  const [isRegistering, setIsRegistering] = useState(true);
  const [userId, setUserId] = useState<string | null>(null);
  const [isInsuranceCovered, setIsInsuranceCovered] = useState(false);
  const [isShareModalVisible, setIsShareModalVisible] = useState(false);
  const [sharePhoneNumber, setSharePhoneNumber] = useState("");
  const [shareRelation, setShareRelation] = useState("");
  const [recentContacts, setRecentContacts] = useState<RecentContact[]>([]);
  const [showRecentContacts, setShowRecentContacts] = useState(false);
  const [loading, setLoading] = useState(false);

  // Document states
  const [medicalReports, setMedicalReports] = useState<any[]>([]);
  const [insuranceDocuments, setInsuranceDocuments] = useState<any[]>([]);

  // Animation refs
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const buttonScale = useRef(new Animated.Value(1)).current;

  // Load user profile if logged in and load recent contacts
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        easing: Easing.out(Easing.exp),
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 5,
        tension: 80,
        useNativeDriver: true,
      }),
    ]).start();

    // Request camera permissions
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Needed', 'Camera permission is needed to take photos');
      }
    })();

    // Load recent contacts from storage
    loadRecentContacts();
  }, []);

  // Load recent contacts from AsyncStorage
  const loadRecentContacts = async () => {
    try {
      const recentContactsData = await AsyncStorage.getItem('recentContacts');
      if (recentContactsData) {
        const parsedContacts = JSON.parse(recentContactsData);
        setRecentContacts(parsedContacts);
      }
    } catch (error) {
      console.error("Error loading recent contacts:", error);
    }
  };

  // Save recent contacts to AsyncStorage
  const saveRecentContact = async (contact: RecentContact) => {
    try {
      // First check if this contact already exists
      const updatedContacts = [...recentContacts];
      const existingIndex = updatedContacts.findIndex(c => c.phoneNumber === contact.phoneNumber);
      
      if (existingIndex !== -1) {
        // Update existing contact
        updatedContacts[existingIndex] = {
          ...contact,
          timestamp: Date.now()
        };
      } else {
        // Add new contact
        updatedContacts.unshift({
          ...contact,
          timestamp: Date.now()
        });
      }
      
      // Keep only the latest 10 contacts
      const limitedContacts = updatedContacts.slice(0, 10);
      
      setRecentContacts(limitedContacts);
      await AsyncStorage.setItem('recentContacts', JSON.stringify(limitedContacts));
    } catch (error) {
      console.error("Error saving recent contact:", error);
    }
  };

  // Fetch user profile from server
  const fetchUserProfile = async (id: string) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/profile/${id}`);
      if (response.data.success) {
        setUserData(response.data.user);
        setIsRegistering(false);
        setIsEditing(false);
      }
    } catch (error) {
      console.error("Error fetching profile:", error);
      Alert.alert(t("Error"), t("Failed to fetch profile. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  // Save user profile
  const handleSave = async () => {
    try {
      setLoading(true);
      // Form validation
      const requiredFields = ['name', 'age', 'gender', 'blood_group', 'date_of_birth'];
      const missingFields = requiredFields.filter(field => !userData[field as keyof UserData]);
      
      if (missingFields.length > 0) {
        Alert.alert(t("Missing Information"), t(`Please fill in the following fields: ${missingFields.join(', ')}`));
        setLoading(false);
        return;
      }

      if (!userData.photo && isRegistering) {
        Alert.alert(t("Missing Photo"), t("Please take a profile photo"));
        setLoading(false);
        return;
      }

      const formData = new FormData();
      
      // Append user data
      Object.keys(userData).forEach(key => {
        if (key === 'photo' && userData.photo) {
          const filename = userData.photo.split('/').pop();
          // Get file extension
          const match = /\.(\w+)$/.exec(filename || '');
          const type = match ? `image/${match[1]}` : 'image/jpeg';
          
          formData.append('photo', {
            uri: userData.photo,
            type: type,
            name: filename || `${userData.name}_profile.jpg`,
          } as any);
        } else if (userData[key as keyof UserData]) {
          formData.append(key, userData[key as keyof UserData] as string);
        }
      });
      
      // If editing, include the user ID
      if (userId) {
        formData.append('_id', userId);
      }
      formData.append('is_insurance_covered', isInsuranceCovered.toString());

      // Append medical reports
      medicalReports.forEach((report, index) => {
        formData.append(`medical_report_${index}`, {
          uri: report.uri,
          type: 'application/pdf', // Adjust the type if needed
          name: report.name,
        } as any);
      });

      // Append insurance documents
      insuranceDocuments.forEach((doc, index) => {
        formData.append(`insurance_document_${index}`, {
          uri: doc.uri,
          type: 'application/pdf', // Adjust the type if needed
          name: doc.name,
        } as any);
      });

      const response = await axios.post(`${API_URL}/profile`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setUserId(response.data.user._id);
        setIsEditing(false);
        setIsRegistering(false);
        Alert.alert(t(isRegistering ? "Registration Successful!" : "Profile Saved Successfully!"));
      }
    } catch (error) {
      console.error("Error saving profile:", error);
      Alert.alert(t("Error"), t("Failed to save profile. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleChange = (key: keyof UserData, value: string) => {
    setUserData({ ...userData, [key]: value });
  };

  // Animation handlers
  const animateButtonPressIn = () => {
    Animated.spring(buttonScale, {
      toValue: 0.95,
      friction: 3,
      tension: 100,
      useNativeDriver: true,
    }).start();
  };

  const animateButtonPressOut = () => {
    Animated.spring(buttonScale, {
      toValue: 1,
      friction: 3,
      tension: 100,
      useNativeDriver: true,
    }).start();
  };

  // Document upload handler
  const handleDocumentUpload = async (type: 'medical' | 'insurance') => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "*/*",
        copyToCacheDirectory: false,
      });

      if (result.type === "success") {
        if (type === 'medical') {
          setMedicalReports([...medicalReports, result]);
        } else if (type === 'insurance') {
          setInsuranceDocuments([...insuranceDocuments, result]);
        }
        Alert.alert("Document Uploaded", result.name);
      }
    } catch (error) {
      console.error("Error picking document:", error);
    }
  };

  // Photo capture handler
  const handleTakePhoto = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.7,
      });
      
      if (!result.canceled) {
        setUserData({ ...userData, photo: result.assets[0].uri });
      }
    } catch (error) {
      console.error("Error taking picture:", error);
      Alert.alert("Error", "Failed to take photo. Please try again.");
    }
  };

  // Open the share modal
  const handleOpenShareModal = () => {
    if (!userData.name) {
      Alert.alert(t("Cannot Share"), t("Please save your profile first."));
      return;
    }
    setIsShareModalVisible(true);
  };

  // Close the share modal
  const handleCloseShareModal = () => {
    setIsShareModalVisible(false);
    setSharePhoneNumber("");
    setShareRelation("");
  };

  // Toggle recent contacts modal
  const toggleRecentContacts = () => {
    setShowRecentContacts(!showRecentContacts);
  };

  // Select a recent contact
  const selectRecentContact = (contact: RecentContact) => {
    setSharePhoneNumber(contact.phoneNumber);
    setShareRelation(contact.relation);
    setShowRecentContacts(false);
  };

  // Share profile with contact
 const handleShareProfile = async () => {
  // Validation
  if (!sharePhoneNumber || sharePhoneNumber.trim() === '') {
    Alert.alert(t("Missing Information"), t("Please enter a phone number."));
    return;
  }

  if (!shareRelation || shareRelation.trim() === '') {
    Alert.alert(t("Missing Information"), t("Please enter your relation to the person."));
    return;
  }

  // Validate phone number format
  const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
  if (!phoneRegex.test(sharePhoneNumber.replace(/\s+/g, ''))) {
    Alert.alert(t("Invalid Phone Number"), t("Please enter a valid phone number."));
    return;
  }

  // Validate required user data
  const requiredFields = ['name', 'age', 'gender', 'blood_group', 'date_of_birth'];
  const missingUserData = requiredFields.filter(field => !userData[field] || userData[field].toString().trim() === '');
  
  if (missingUserData.length > 0) {
    Alert.alert(
      t("Incomplete Profile"), 
      t(`Please complete the following fields: ${missingUserData.join(', ')}`)
    );
    return;
  }

  // Confirmation dialog for sharing sensitive information
  Alert.alert(
    t("Confirm Share"),
    t("You are about to share sensitive medical information via SMS. Do you want to continue?"),
    [
      {
        text: t("Cancel"),
        style: "cancel"
      },
      {
        text: t("Share"),
        onPress: async () => {
          try {
            setLoading(true);
            
            // Clean phone number
            const cleanedPhoneNumber = sharePhoneNumber.replace(/\s+/g, '');
            
            // Prepare data to send
            const dataToSend = {
              userData: {
                name: userData.name?.trim(),
                age: userData.age,
                gender: userData.gender?.trim(),
                blood_group: userData.blood_group?.trim(),
                date_of_birth: userData.date_of_birth?.trim(),
                medical_conditions: userData.medical_conditions?.trim() || "None"
              },
              phoneNumber: cleanedPhoneNumber,
              relation: shareRelation.trim(),
              medicalReports: medicalReports.length > 0,
              insuranceDocuments: insuranceDocuments.length > 0
            };
            
            const response = await axios.post(`${API_URL}/getno`, dataToSend, {
              timeout: 15000, // 15 second timeout
              headers: {
                'Content-Type': 'application/json'
              }
            });

            if (response.data.success) {
              // Save to recent contacts
              await saveRecentContact({
                name: userData.name,
                phoneNumber: cleanedPhoneNumber,
                relation: shareRelation.trim(),
                timestamp: Date.now(),
                messageSid: response.data.messageSid
              });
              
              Alert.alert(
                t("Success"), 
                t("Emergency contact information shared successfully!")
              );
              handleCloseShareModal();
            } else {
              Alert.alert(
                t("Error"), 
                response.data.message || t("Failed to share profile. Please try again.")
              );
            }
          } catch (error) {
            console.error("Error sharing profile:", error);
            
            let errorMessage = t("Failed to share profile. Please try again.");
            
            if (error.response?.data?.message) {
              errorMessage = error.response.data.message;
            } else if (error.code === 'ECONNABORTED') {
              errorMessage = t("Request timeout. Please check your connection and try again.");
            } else if (error.code === 'NETWORK_ERROR') {
              errorMessage = t("Network error. Please check your internet connection.");
            }
            
            Alert.alert(t("Error"), errorMessage);
          } finally {
            setLoading(false);
          }
        }
      }
    ]
  );
};

  // Render a recent contact item
  const renderRecentContactItem = ({ item }: { item: RecentContact }) => (
    <TouchableOpacity
      style={styles.recentContactItem}
      onPress={() => selectRecentContact(item)}
    >
      <View style={styles.contactIconContainer}>
        <Ionicons name="person-circle-outline" size={36} color="#4a86ff" />
      </View>
      <View style={styles.contactInfo}>
        <Text style={styles.contactName}>{item.phoneNumber}</Text>
        <Text style={styles.contactRelation}>{item.relation}</Text>
        <Text style={styles.contactTimestamp}>
          {new Date(item.timestamp).toLocaleDateString()}
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={{ flex: 1 }}>
      <ScrollView contentContainerStyle={styles.container}>
        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#4a86ff" />
          </View>
        )}

        <Animated.Text style={[styles.header, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
          {isRegistering ? t("Register Profile") : t("User Profile")}
        </Animated.Text>

        {userData.photo ? (
          <Image source={{ uri: userData.photo }} style={styles.profilePhoto} />
        ) : (
          <Image source={require('@/assets/images/profile-icon.gif')} style={styles.profileIcon} />
        )}

        {isEditing && (
          <TouchableOpacity style={styles.photoButton} onPress={handleTakePhoto}>
            <Text style={styles.photoButtonText}>{userData.photo ? t("Change Photo") : t("Take Profile Photo")}</Text>
          </TouchableOpacity>
        )}

        <Animated.View style={[styles.card, { opacity: fadeAnim }]}>
          {Object.keys(userData)
            .filter(key => key !== 'photo' && key !== '_id')
            .map((key) => (
              <View style={styles.inputContainer} key={key}>
                <Text style={styles.label}>{t(key.replace(/([A-Z])/g, " $1").trim())}</Text>
                <TextInput
                  style={[styles.input, !isEditing && styles.disabledInput]}
                  placeholder={t(`Enter ${key}`)}
                  placeholderTextColor="#888"
                  value={userData[key as keyof UserData] as string}
                  onChangeText={(text) => handleChange(key as keyof UserData, text)}
                  editable={isEditing}
                />
              </View>
            ))}
        </Animated.View>

        {isEditing && (
          <View style={styles.insuranceContainer}>
            <Text style={styles.insuranceLabel}>{t("Is Insurance Covered?")}</Text>
            <Switch
              value={isInsuranceCovered}
              onValueChange={setIsInsuranceCovered}
              trackColor={{ false: "#767577", true: "#81b0ff" }}
              thumbColor={isInsuranceCovered ? "#f5dd4b" : "#f4f3f4"}
            />
          </View>
        )}

        <View style={styles.buttonsContainer}>
          <Animated.View style={{ transform: [{ scale: buttonScale }], flex: 1, marginRight: 5 }}>
            {isEditing ? (
              <TouchableOpacity
                style={styles.saveButton}
                onPress={handleSave}
                onPressIn={animateButtonPressIn}
                onPressOut={animateButtonPressOut}
              >
                <Text style={styles.saveText}>{isRegistering ? t("Register") : t("Save Profile")}</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={styles.editButton}
                onPress={handleEdit}
                onPressIn={animateButtonPressIn}
                onPressOut={animateButtonPressOut}
              >
                <Text style={styles.editText}>{t("Edit Profile")}</Text>
              </TouchableOpacity>
            )}
          </Animated.View>

          {/* Share button - visible when profile is saved */}
          {!isEditing && userData.name && (
            <Animated.View style={{ transform: [{ scale: buttonScale }], flex: 1, marginLeft: 5 }}>
              <TouchableOpacity
                style={styles.shareButton}
                onPress={handleOpenShareModal}
                onPressIn={animateButtonPressIn}
                onPressOut={animateButtonPressOut}
              >
                <View style={styles.buttonContent}>
                  <Ionicons name="share-social" size={18} color="#fff" style={styles.buttonIcon} />
                  <Text style={styles.shareText}>{t("Share Profile")}</Text>
                </View>
              </TouchableOpacity>
            </Animated.View>
          )}
        </View>

        {/* Document Upload Buttons */}
        <TouchableOpacity style={styles.uploadButton} onPress={() => handleDocumentUpload('medical')}>
          <Text style={styles.uploadText}>{t("Upload Medical Reports")}</Text>
        </TouchableOpacity>

        {isInsuranceCovered && (
          <TouchableOpacity style={styles.uploadButton} onPress={() => handleDocumentUpload('insurance')}>
            <Text style={styles.uploadText}>{t("Upload Insurance Documents")}</Text>
          </TouchableOpacity>
        )}

        {/* Display uploaded documents */}
        {medicalReports.length > 0 && (
          <View style={styles.documentsContainer}>
            <Text style={styles.documentTitle}>{t("Medical Reports:")}</Text>
            {medicalReports.map((doc, index) => (
              <Text key={index} style={styles.documentName}>{doc.name}</Text>
            ))}
          </View>
        )}

        {insuranceDocuments.length > 0 && (
          <View style={styles.documentsContainer}>
            <Text style={styles.documentTitle}>{t("Insurance Documents:")}</Text>
            {insuranceDocuments.map((doc, index) => (
              <Text key={index} style={styles.documentName}>{doc.name}</Text>
            ))}
          </View>
        )}

        {/* Share Profile Modal */}
        <Modal
          visible={isShareModalVisible}
          transparent
          animationType="slide"
          onRequestClose={handleCloseShareModal}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>{t("Share Profile")}</Text>
              
              <View style={styles.inputContainer}>
                <Text style={styles.modalLabel}>{t("Phone Number")}</Text>
                <TextInput
                  style={styles.modalInput}
                  placeholder={t("Enter recipient's phone number with +91")}
                  placeholderTextColor="#888"
                  value={sharePhoneNumber}
                  onChangeText={setSharePhoneNumber}
                  keyboardType="phone-pad"
                />
              </View>
              
              <View style={styles.inputContainer}>
                <Text style={styles.modalLabel}>{t("Relation")}</Text>
                <TextInput
                  style={styles.modalInput}
                  placeholder={t("Enter your relation to this person")}
                  placeholderTextColor="#888"
                  value={shareRelation}
                  onChangeText={setShareRelation}
                />
              </View>

              {/* Recent contacts button */}
              {recentContacts.length > 0 && (
                <TouchableOpacity 
                  style={styles.recentContactsButton}
                  onPress={toggleRecentContacts}
                >
                  <Ionicons name="time-outline" size={18} color="#4a86ff" />
                  <Text style={styles.recentContactsText}>{t("Recent Contacts")}</Text>
                </TouchableOpacity>
              )}

              <View style={styles.modalButtonsContainer}>
                <TouchableOpacity
                  style={styles.cancelButton}
                  onPress={handleCloseShareModal}
                >
                  <Text style={styles.cancelButtonText}>{t("Cancel")}</Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={styles.confirmButton}
                  onPress={handleShareProfile}
                >
                  <Text style={styles.confirmButtonText}>{t("Share")}</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>

        {/* Recent Contacts Modal */}
        <Modal
          visible={showRecentContacts}
          transparent
          animationType="slide"
          onRequestClose={() => setShowRecentContacts(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>{t("Recent Contacts")}</Text>
              
              <FlatList
                data={recentContacts}
                renderItem={renderRecentContactItem}
                keyExtractor={(item, index) => `contact-${index}`}
                style={styles.contactsList}
              />
              
              <TouchableOpacity
                style={[styles.cancelButton, { marginTop: 10 }]}
                onPress={() => setShowRecentContacts(false)}
              >
                <Text style={styles.cancelButtonText}>{t("Close")}</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { 
    flexGrow: 1, 
    padding: 20, 
    backgroundColor: "#25292e" 
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  header: { 
    fontSize: 24, 
    color: "#fff", 
    fontWeight: "bold", 
    textAlign: "center", 
    marginBottom: 20 
  },
  profileIcon: { 
    width: 100, 
    height: 100, 
    alignSelf: "center", 
    marginBottom: 10 
  },
  profilePhoto: { 
    width: 150, 
    height: 150, 
    borderRadius: 75, 
    alignSelf: "center", 
    marginBottom: 10 
  },
  card: { 
    backgroundColor: "#333",
    width: "100%",
    borderRadius: 10,
    padding: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  inputContainer: { 
    marginBottom: 12 
  },
  label: {
    color: "#aaa",
    fontSize: 16,
    fontWeight: "600",
    textTransform: "capitalize",
  },
  input: { 
    borderWidth: 1, 
    borderColor: "#CCC", 
    borderRadius: 5, 
    padding: 8, 
    backgroundColor: "#FFF" 
  },
  disabledInput: { 
    backgroundColor: "#E0E0E0" 
  },
  buttonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  editButton: { 
    backgroundColor: "#007BFF", 
    padding: 12, 
    borderRadius: 10, 
    alignItems: "center",
    justifyContent: "center",
  },
  editText: { 
    color: "#FFF", 
    fontWeight: "bold" 
  },
  shareButton: {
    backgroundColor: "#4a86ff",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  buttonIcon: {
    marginRight: 5,
  },
  shareText: {
    color: "#fff",
    fontWeight: "bold",
  },
  documentsContainer: {
    marginTop: 20,
    width: "90%",
  },
  documentName: {
    color: "#fff",
    fontSize: 16,
    marginBottom: 5,
  },
  documentTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 5,
  },
  uploadText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  uploadButton: {
    backgroundColor: "#007BFF",
    width: "100%",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 10,
  },
  saveButton: {
    backgroundColor: "#00b894",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  saveText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  photoButton: {
    backgroundColor: "#e67e22",
    padding: 10,
    borderRadius: 8,
    alignItems: "center",
    marginBottom: 15,
    width: "60%",
    alignSelf: "center",
  },
  photoButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  insuranceContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 10,
    marginBottom: 15,
    marginTop: 10,
  },
  insuranceLabel: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '85%',
    backgroundColor: '#333',
    borderRadius: 15,
    padding: 20,
    maxHeight: '80%',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 20,
  },
  modalLabel: {
    color: "#aaa",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 5,
  },
  modalInput: {
    borderWidth: 1,
    borderColor: "#CCC",
    borderRadius: 5,
    padding: 10,
    backgroundColor: "#FFF",
    marginBottom: 15,
  },
  modalButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  cancelButton: {
    backgroundColor: '#6c757d',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    flex: 1,
    marginRight: 5,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  confirmButton: {
    backgroundColor: '#4a86ff',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    flex: 1,
    marginLeft: 5,
    alignItems: 'center',
  },
  confirmButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  // Recent contacts styles
  recentContactsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 15,
    padding: 10,
    backgroundColor: 'rgba(74, 134, 255, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(74, 134, 255, 0.3)',
  },
  recentContactsText: {
    marginLeft: 8,
    color: '#4a86ff',
    fontWeight: '600',
  },
  contactsList: {
    maxHeight: 250,
  },
  recentContactItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  contactIconContainer: {
    marginRight: 10,
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  contactRelation: {
    color: '#aaa',
    fontSize: 14,
  },
  contactTimestamp: {
    color: '#777',
    fontSize: 12,
  },
  documentsContainer: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 5,
  },
  documentsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  documentItem: {
    fontSize: 16,
    marginBottom: 5,
    color: '#333',
  },
});
