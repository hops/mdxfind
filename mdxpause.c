/*
 * mdxpause.c --List and pause/resume mdxfind instances
 *
 * Usage: mdxpause pause    --list instances, select one to pause
 *        mdxpause resume   --list instances, select one to resume
 *
 * Linux:   sends SIGUSR1 (pause) / SIGUSR2 (resume)
 * Windows: signals named events Global\mdxfind_pause_<pid> / _resume_<pid>
 *
 * Portable: compiles on Linux, macOS, FreeBSD, and Windows (MinGW).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32

#include <windows.h>
#include <tlhelp32.h>

typedef struct { DWORD pid; char cmdline[256]; } proc_t;

static int list_mdxfind(proc_t *procs, int max)
{
    int count = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 pe;

    if (snap == INVALID_HANDLE_VALUE)
        return 0;

    pe.dwSize = sizeof(pe);
    if (Process32First(snap, &pe)) {
        do {
            if (_stricmp(pe.szExeFile, "mdxfind.exe") == 0 && count < max) {
                procs[count].pid = pe.th32ProcessID;
                snprintf(procs[count].cmdline, sizeof(procs[count].cmdline),
                         "%s (parent PID %u)", pe.szExeFile, pe.th32ParentProcessID);
                count++;
            }
        } while (Process32Next(snap, &pe));
    }
    CloseHandle(snap);
    return count;
}

static int send_signal(DWORD pid, const char *action)
{
    char name[64];
    HANDLE h;

    snprintf(name, sizeof(name), "Global\\mdxfind_%s_%u", action, (unsigned)pid);
    h = OpenEvent(EVENT_MODIFY_STATE, FALSE, name);
    if (!h) {
        fprintf(stderr, "  Cannot signal PID %u --not responding or no event\n", (unsigned)pid);
        return 1;
    }
    SetEvent(h);
    CloseHandle(h);
    return 0;
}

#else /* POSIX: Linux, macOS, FreeBSD */

#include <signal.h>
#include <dirent.h>
#include <unistd.h>
#include <ctype.h>

#ifdef __APPLE__
#include <libproc.h>
#include <sys/sysctl.h>
#endif

typedef struct { pid_t pid; char cmdline[256]; } proc_t;

static int list_mdxfind(proc_t *procs, int max)
{
    int count = 0;

#ifdef __APPLE__
    /* macOS: use proc_listallpids + proc_pidpath */
    int npids;
    pid_t *pids;
    pid_t mypid = getpid();

    npids = proc_listallpids(NULL, 0);
    if (npids <= 0) return 0;

    pids = malloc(sizeof(pid_t) * npids);
    if (!pids) return 0;

    npids = proc_listallpids(pids, sizeof(pid_t) * npids);

    for (int i = 0; i < npids && count < max; i++) {
        char path[PROC_PIDPATHINFO_MAXSIZE];
        char args[256];

        if (pids[i] == mypid) continue;
        if (proc_pidpath(pids[i], path, sizeof(path)) <= 0) continue;

        /* Check if the executable name contains "mdxfind" */
        char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (strstr(base, "mdxfind") == NULL) continue;

        /* Get arguments via procargs */
        int mib[3] = { CTL_KERN, KERN_PROCARGS2, pids[i] };
        size_t arglen = sizeof(args);
        if (sysctl(mib, 3, args, &arglen, NULL, 0) == 0 && arglen > sizeof(int)) {
            /* Skip argc, find the executable path and args */
            int nargs;
            memcpy(&nargs, args, sizeof(int));
            char *p = args + sizeof(int);
            char *end = args + arglen;
            /* Skip exec path */
            while (p < end && *p != 0) p++;
            while (p < end && *p == 0) p++;
            /* Build cmdline from remaining args */
            char cmdline[256] = "";
            int cmdpos = 0;
            for (int a = 0; a < nargs && p < end; a++) {
                int len = strlen(p);
                if (cmdpos + len + 2 < (int)sizeof(cmdline)) {
                    if (cmdpos > 0) cmdline[cmdpos++] = ' ';
                    memcpy(cmdline + cmdpos, p, len);
                    cmdpos += len;
                }
                p += len + 1;
            }
            cmdline[cmdpos] = 0;
            strncpy(procs[count].cmdline, cmdline, sizeof(procs[count].cmdline) - 1);
        } else {
            strncpy(procs[count].cmdline, path, sizeof(procs[count].cmdline) - 1);
        }

        procs[count].pid = pids[i];
        count++;
    }
    free(pids);

#else
    /* Linux/FreeBSD: read /proc */
    DIR *d = opendir("/proc");
    struct dirent *de;
    pid_t mypid = getpid();

    if (!d) return 0;

    while ((de = readdir(d)) != NULL && count < max) {
        char path[128], buf[256];
        FILE *f;
        int n, pid;

        if (!isdigit((unsigned char)de->d_name[0])) continue;
        pid = atoi(de->d_name);
        if (pid == mypid) continue;

        snprintf(path, sizeof(path), "/proc/%s/cmdline", de->d_name);
        f = fopen(path, "r");
        if (!f) continue;
        n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        if (n <= 0) continue;
        buf[n] = 0;

        if (strstr(buf, "mdxfind") == NULL) continue;

        procs[count].pid = pid;
        /* Replace nulls with spaces for display */
        for (int i = 0; i < n - 1; i++)
            if (buf[i] == 0) buf[i] = ' ';
        strncpy(procs[count].cmdline, buf, sizeof(procs[count].cmdline) - 1);
        count++;
    }
    closedir(d);
#endif

    return count;
}

static int send_signal(pid_t pid, const char *action)
{
    int sig = (action[0] == 'p') ? SIGUSR1 : SIGUSR2;

    if (kill(pid, sig) != 0) {
        perror("  kill");
        return 1;
    }
    return 0;
}

#endif /* _WIN32 */

int main(int argc, char **argv)
{
    proc_t procs[64];
    int count, sel;
    const char *action;
    char input[16];

    if (argc < 2 || (argv[1][0] != 'p' && argv[1][0] != 'r')) {
        fprintf(stderr, "Usage: %s <pause|resume> [pid]\n", argv[0]);
        return 1;
    }

    action = (argv[1][0] == 'p') ? "pause" : "resume";

    /* Non-interactive: direct PID on command line */
    if (argc >= 3) {
#ifdef _WIN32
        DWORD pid = (DWORD)atoi(argv[2]);
#else
        pid_t pid = (pid_t)atoi(argv[2]);
#endif
        printf("  %s PID %d... ", action, (int)pid);
        if (send_signal(pid, action) == 0)
            printf("OK\n");
        else
            return 1;
        return 0;
    }

    /* Interactive: list and select */
    count = list_mdxfind(procs, 64);

    if (count == 0) {
        printf("No mdxfind instances found.\n");
        return 1;
    }

    if (count == 1) {
        /* Single instance -- act on it directly */
        printf("  %s PID %d (%s)... ", action, (int)procs[0].pid, procs[0].cmdline);
        if (send_signal(procs[0].pid, action) == 0)
            printf("OK\n");
        return 0;
    }

    printf("mdxfind instances:\n\n");
    printf("  #  PID      Command\n");
    printf("  -  -------  -------\n");
    for (int i = 0; i < count; i++)
        printf("  %d  %-7d  %s\n", i + 1, (int)procs[i].pid, procs[i].cmdline);
    printf("  0           All instances\n");

    printf("\nSelect instance to %s [1-%d, 0=all]: ", action, count);
    fflush(stdout);

    if (!fgets(input, sizeof(input), stdin))
        return 1;
    sel = atoi(input);

    if (sel == 0) {
        for (int i = 0; i < count; i++) {
            printf("  %s PID %d... ", action, (int)procs[i].pid);
            if (send_signal(procs[i].pid, action) == 0)
                printf("OK\n");
        }
    } else if (sel >= 1 && sel <= count) {
        printf("  %s PID %d... ", action, (int)procs[sel - 1].pid);
        if (send_signal(procs[sel - 1].pid, action) == 0)
            printf("OK\n");
    } else {
        printf("Invalid selection.\n");
        return 1;
    }

    return 0;
}
