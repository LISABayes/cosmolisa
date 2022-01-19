import numpy as np
import matplotlib.pyplot as plt

snr_M101 = np.genfromtxt("/home/laghi/src/cosmolisa/cosmolisa/scripts/snr_list_M101_DE_1.txt")
snr_M106 = np.genfromtxt("/home/laghi/src/cosmolisa/cosmolisa/scripts/snr_list_M106_DE_1.txt")
snr_M105 = np.genfromtxt("/home/laghi/src/cosmolisa/cosmolisa/scripts/snr_list_M105_DE_1.txt")

M1_median = np.median(snr_M101)
M1_percentile_70 = np.percentile(snr_M101, 70)
M1_percentile_75 = np.percentile(snr_M101, 75)
M1_percentile_80 = np.percentile(snr_M101, 80)
M1_percentile_85 = np.percentile(snr_M101, 85)
M1_percentile_90 = np.percentile(snr_M101, 90)

snr_M101_top_50 = [f for f in snr_M101 if (f >= M1_median)]
snr_M101_top_30 = [f for f in snr_M101 if (f >= M1_percentile_70)]
snr_M101_top_25 = [f for f in snr_M101 if (f >= M1_percentile_75)]
snr_M101_top_20 = [f for f in snr_M101 if (f >= M1_percentile_80)]
snr_M101_top_15 = [f for f in snr_M101 if (f >= M1_percentile_85)]
snr_M101_top_10 = [f for f in snr_M101 if (f >= M1_percentile_90)]

print("M1 top 50%:", len(snr_M101_top_50))
print("M1 top 30%:", len(snr_M101_top_30))
print("M1 top 25%:", len(snr_M101_top_25))
print("M1 top 20%:", len(snr_M101_top_20))
print("M1 top 15%:", len(snr_M101_top_15))
print("M1 top 10%:", len(snr_M101_top_10))
print("M1 top 25 threshold", M1_percentile_75)

M6_median = np.median(snr_M106)
M6_percentile_70 = np.percentile(snr_M106, 70)
M6_percentile_75 = np.percentile(snr_M106, 75)
M6_percentile_80 = np.percentile(snr_M106, 80)
M6_percentile_85 = np.percentile(snr_M106, 85)
M6_percentile_90 = np.percentile(snr_M106, 90)

snr_M106_top_50 = [f for f in snr_M106 if (f >= M6_median)]
snr_M106_top_30 = [f for f in snr_M106 if (f >= M6_percentile_70)]
snr_M106_top_25 = [f for f in snr_M106 if (f >= M6_percentile_75)]
snr_M106_top_20 = [f for f in snr_M106 if (f >= M6_percentile_80)]
snr_M106_top_15 = [f for f in snr_M106 if (f >= M6_percentile_85)]
snr_M106_top_10 = [f for f in snr_M106 if (f >= M6_percentile_90)]

print("")
print("M6 top 50%:", len(snr_M106_top_50))
print("M6 top 30%:", len(snr_M106_top_30))
print("M6 top 25%:", len(snr_M106_top_25))
print("M6 top 20%:", len(snr_M106_top_20))
print("M6 top 15%:", len(snr_M106_top_15))
print("M6 top 10%:", len(snr_M106_top_10))
print("M6 top 25 threshold", M6_percentile_75)

M5_median = np.median(snr_M106)
M5_percentile_70 = np.percentile(snr_M105, 70)
M5_percentile_75 = np.percentile(snr_M105, 75)
M5_percentile_80 = np.percentile(snr_M105, 80)
M5_percentile_85 = np.percentile(snr_M105, 85)
M5_percentile_90 = np.percentile(snr_M105, 90)

snr_M105_top_50 = [f for f in snr_M105 if (f >= M5_median)]
snr_M105_top_30 = [f for f in snr_M105 if (f >= M5_percentile_70)]
snr_M105_top_25 = [f for f in snr_M105 if (f >= M5_percentile_75)]
snr_M105_top_20 = [f for f in snr_M105 if (f >= M5_percentile_80)]
snr_M105_top_15 = [f for f in snr_M105 if (f >= M5_percentile_85)]
snr_M105_top_10 = [f for f in snr_M105 if (f >= M5_percentile_90)]

print("")
print("M5 top 50%:", len(snr_M105_top_50))
print("M5 top 30%:", len(snr_M105_top_30))
print("M5 top 25%:", len(snr_M105_top_25))
print("M5 top 20%:", len(snr_M105_top_20))
print("M5 top 15%:", len(snr_M105_top_15))
print("M5 top 10%:", len(snr_M105_top_10))
print("M5 top 25 threshold", M5_percentile_75)

# M1 should have SNRmin 77 to get the 50 loudest events
# M6 should have SNRmin 117 to get the 50 loudest events

plt.figure()
plt.hist(snr_M101, bins=50, label='M101', histtype='step')
plt.hist(snr_M106, bins=50, label='M106', histtype='step')
plt.hist(snr_M105, bins=50, label='M105', histtype='step')
plt.legend()
plt.ylabel("Number of EMRIs")
plt.xlabel("SNR")

plt.savefig("M1_M5_M6_DE_snr.pdf", bbox_inches='tight')